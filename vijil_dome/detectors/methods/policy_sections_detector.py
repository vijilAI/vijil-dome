# Copyright 2025 Vijil, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# vijil and vijil-dome are trademarks of Vijil Inc.

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union, Tuple

from vijil_dome.detectors import (
    DetectionCategory,
    DetectionMethod,
    register_method,
    POLICY_SECTIONS,
)
from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import PolicyGptOssSafeguard
from vijil_dome.utils.policy_loader import (
    load_policy_sections_from_s3,
    validate_policy_json,
)
from vijil_dome.utils.faiss_loader import (
    load_faiss_index_from_s3,
    load_section_ids_from_s3,
)
from vijil_dome.embeds import EmbeddingsItem
from vijil_dome.embeds.embedder import Embedder

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Policy, POLICY_SECTIONS)
class PolicySectionsDetector(DetectionMethod):
    """
    Policy sections detector that loads policy sections from S3 or accepts them directly,
    creates multiple PolicyGptOssSafeguard instances (one per section), and runs them
    in parallel batches with fast fail.

    This detector is optimized for policies that are split into 400-600 token sections
    for optimal performance. Each section is evaluated independently, and the detector
    fails fast on the first violation detected.

    Example with S3:
        detector = PolicySectionsDetector(
            policy_s3_bucket="my-bucket",
            policy_s3_key="teams/team-123/policies/policy-456/sections.json",
            applies_to="input",
            max_parallel_sections=10,
            model_name="openai/gpt-oss-120b",
            reasoning_effort="medium"
        )

    Example with direct sections:
        detector = PolicySectionsDetector(
            policy_sections=[
                {
                    "section_id": "section-1",
                    "content": "# Policy\n\n## INSTRUCTIONS\n...",
                    "applies_to": ["input"],
                }
            ],
            applies_to="input",
            model_name="openai/gpt-oss-120b",
        )
    """

    def __init__(
        self,
        # S3 loading (mutually exclusive with policy_sections)
        policy_s3_bucket: Optional[str] = None,
        policy_s3_key: Optional[str] = None,
        # Direct sections (mutually exclusive with S3 params)
        policy_sections: Optional[List[Dict[str, Any]]] = None,
        # Filtering
        applies_to: Union[str, List[str]] = "input",
        # Rate limiting
        max_parallel_sections: Optional[int] = None,
        # PolicyGptOssSafeguard params
        model_name: str = "openai/gpt-oss-120b",
        reasoning_effort: str = "medium",
        hub_name: str = "openai",
        timeout: Optional[int] = 60,
        max_retries: Optional[int] = 3,
        api_key: Optional[str] = None,
        # S3 auth params (optional)
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        # RAG parameters (optional)
        use_rag: bool = False,
        faiss_s3_key: Optional[str] = None,
        section_ids_s3_key: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_engine: str = "SentenceTransformers",
    ):
        """
        Initialize the policy sections detector.

        Args:
            policy_s3_bucket: S3 bucket name (required if policy_sections not provided)
            policy_s3_key: S3 object key (required if policy_sections not provided)
            policy_sections: Direct list of section dicts (required if S3 params not provided)
            applies_to: Filter sections by applies_to - "input", "output", or ["input", "output"] (default: "input")
            max_parallel_sections: Maximum number of sections to run in parallel (default: None = no limit)
            model_name: LLM model identifier (default: "openai/gpt-oss-120b")
            reasoning_effort: Reasoning depth - "low", "medium", "high" (default: "medium")
            hub_name: LLM hub to use (default: "nebius")
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts (default: 3)
            api_key: API key for the hub (optional, uses env var if not provided)
            aws_access_key_id: AWS access key (optional, uses boto3 defaults if not provided)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
            region_name: AWS region (optional, uses boto3 defaults if not provided)
            cache_dir: Local cache directory (optional, defaults to ~/.cache/vijil-dome/policies/)
            use_rag: Enable RAG-based retrieval to select relevant sections (default: False)
            faiss_s3_key: S3 key for faiss.index file (required if use_rag=True)
            section_ids_s3_key: S3 key for section_ids.json file (required if use_rag=True)
            top_k: Number of top sections to retrieve via RAG (default: 5)
            similarity_threshold: Minimum similarity score to include a section (default: 0.0)
            embedding_model: Embedding model for query encoding (default: "all-MiniLM-L6-v2")
            embedding_engine: Embedding engine - "OpenAI", "FastEmbed", or "SentenceTransformers" (default: "SentenceTransformers")

        Raises:
            ValueError: If neither S3 params nor policy_sections provided, or both provided
            ValueError: If sections structure is invalid
            ValueError: If use_rag=True but FAISS files not provided
        """
        # Validate parameters
        has_s3, has_sections = self._validate_params(
            policy_s3_bucket, policy_s3_key, policy_sections, use_rag,
            faiss_s3_key, section_ids_s3_key
        )
        
        # Initialize RAG components if enabled
        self._initialize_rag_components(
            use_rag, has_s3, policy_s3_bucket, faiss_s3_key, section_ids_s3_key,
            cache_dir, aws_access_key_id, aws_secret_access_key,
            aws_session_token, region_name, embedding_model, embedding_engine,
            top_k, similarity_threshold
        )
        
        # Load and filter policy sections
        policy_data, filtered_sections = self._load_and_filter_sections(
            has_s3, policy_s3_bucket, policy_s3_key, policy_sections,
            applies_to, cache_dir, aws_access_key_id, aws_secret_access_key,
            aws_session_token, region_name
        )
        
        # Create detectors for each section
        self._create_detectors(
            filtered_sections, policy_data, model_name, reasoning_effort,
            hub_name, timeout, max_retries, api_key
        )
        
        # Populate FAISS index with items if RAG is enabled
        if self.use_rag and self.faiss_index is not None:
            self._populate_faiss_items(filtered_sections)
        
        self.max_parallel_sections = max_parallel_sections
        self.blocked_response_string = f"Method:{POLICY_SECTIONS}"
    
    def _validate_params(
        self,
        policy_s3_bucket: Optional[str],
        policy_s3_key: Optional[str],
        policy_sections: Optional[List[Dict[str, Any]]],
        use_rag: bool,
        faiss_s3_key: Optional[str],
        section_ids_s3_key: Optional[str],
    ) -> Tuple[bool, bool]:
        """Validate input parameters and return flags for S3/sections presence."""
        has_s3 = policy_s3_bucket is not None and policy_s3_key is not None
        has_sections = policy_sections is not None

        if not has_s3 and not has_sections:
            raise ValueError(
                "Either policy_s3_bucket/policy_s3_key or policy_sections must be provided"
            )
        if has_s3 and has_sections:
            raise ValueError(
                "Cannot specify both S3 params and policy_sections"
            )
        
        # Validate RAG parameters
        if use_rag:
            if not has_s3:
                raise ValueError(
                    "RAG requires S3 loading. Provide policy_s3_bucket and policy_s3_key."
                )
            if faiss_s3_key is None or section_ids_s3_key is None:
                raise ValueError(
                    "RAG requires both faiss_s3_key and section_ids_s3_key to be provided"
                )
        
        return has_s3, has_sections
    
    def _initialize_rag_components(
        self,
        use_rag: bool,
        has_s3: bool,
        policy_s3_bucket: Optional[str],
        faiss_s3_key: Optional[str],
        section_ids_s3_key: Optional[str],
        cache_dir: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
        region_name: Optional[str],
        embedding_model: str,
        embedding_engine: str,
        top_k: int,
        similarity_threshold: float,
    ) -> None:
        """Initialize RAG components if enabled."""
        self.use_rag = use_rag
        self.faiss_index = None
        self.section_ids_mapping: Dict[str, str] = {}
        self.embedder = None
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        if not use_rag or not has_s3:
            return
        
        try:
            # Load FAISS index
            faiss_index_path = load_faiss_index_from_s3(
                bucket=policy_s3_bucket,
                key=faiss_s3_key,
                cache_dir=cache_dir,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            
            # Load section_ids mapping
            self.section_ids_mapping = load_section_ids_from_s3(
                bucket=policy_s3_bucket,
                key=section_ids_s3_key,
                cache_dir=cache_dir,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            
            # Initialize embedder
            self.embedder = Embedder(
                embedding_model=embedding_model,
                embedding_engine=embedding_engine,
            )
            
            # Load FAISS index
            try:
                from vijil_dome.embeds.index.faiss_index import FaissEmbeddingsIndex
                self.faiss_index = FaissEmbeddingsIndex(embedder=None)
                self.faiss_index.load_from_file(faiss_index_path)
                
                # Check dimension compatibility
                index_dim = self.faiss_index.get_embedding_size()
                embedder_dim = self.embedder.get_embedding_size()
                
                if index_dim != embedder_dim:
                    suggested_model = self._suggest_embedding_model(index_dim)
                    logger.warning(
                        "FAISS index dimension mismatch",
                        extra={
                            "index_dim": index_dim,
                            "embedder_dim": embedder_dim,
                            "embedding_model": embedding_model,
                            "suggested_model": suggested_model,
                        }
                    )
                    self.use_rag = False
                    self.faiss_index = None
                else:
                    self.faiss_index._embedder = self.embedder
                    logger.info(
                        "Loaded FAISS index",
                        extra={
                            "dimension": index_dim,
                            "section_count": len(self.section_ids_mapping),
                        }
                    )
            except ImportError:
                logger.warning(
                    "faiss-cpu not installed. RAG will be disabled. "
                    "Install with: pip install faiss-cpu"
                )
                self.use_rag = False
            except Exception as e:
                logger.warning(
                    "Failed to load FAISS index",
                    extra={"error": str(e), "faiss_path": faiss_index_path}
                )
                self.use_rag = False
        except Exception as e:
            logger.warning(
                "Failed to initialize RAG",
                extra={
                    "error": str(e),
                    "bucket": policy_s3_bucket,
                    "faiss_key": faiss_s3_key,
                    "section_ids_key": section_ids_s3_key,
                }
            )
            self.use_rag = False
    
    def _suggest_embedding_model(self, index_dim: int) -> Optional[str]:
        """Suggest embedding model based on FAISS index dimension."""
        if index_dim == 1536:
            return "text-embedding-ada-002"
        elif index_dim == 384:
            return "all-MiniLM-L6-v2"
        elif index_dim == 768:
            return "all-mpnet-base-v2"
        return None
    
    def _load_and_filter_sections(
        self,
        has_s3: bool,
        policy_s3_bucket: Optional[str],
        policy_s3_key: Optional[str],
        policy_sections: Optional[List[Dict[str, Any]]],
        applies_to: Union[str, List[str]],
        cache_dir: Optional[str],
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
        region_name: Optional[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Load policy sections and filter by applies_to."""
        # Load sections
        if has_s3:
            policy_data = load_policy_sections_from_s3(
                bucket=policy_s3_bucket,
                key=policy_s3_key,
                cache_dir=cache_dir,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            sections = policy_data.get("sections", [])
        else:
            if not isinstance(policy_sections, list) or len(policy_sections) == 0:
                raise ValueError("policy_sections must be a non-empty list")
            policy_data = {"sections": policy_sections}
            validate_policy_json(policy_data)
            sections = policy_sections

        # Normalize applies_to to list
        if isinstance(applies_to, str):
            applies_to_list = [applies_to]
        else:
            applies_to_list = applies_to

        # Filter sections by applies_to
        filtered_sections = []
        for section in sections:
            section_applies_to = section.get("applies_to", [])
            if not isinstance(section_applies_to, list):
                section_applies_to = [section_applies_to]
            if any(applies in section_applies_to for applies in applies_to_list):
                filtered_sections.append(section)

        if len(filtered_sections) == 0:
            raise ValueError(
                f"No sections found matching applies_to={applies_to_list}"
            )
        
        return policy_data, filtered_sections
    
    def _create_detectors(
        self,
        filtered_sections: List[Dict[str, Any]],
        policy_data: Dict[str, Any],
        model_name: str,
        reasoning_effort: str,
        hub_name: str,
        timeout: Optional[int],
        max_retries: Optional[int],
        api_key: Optional[str],
    ) -> None:
        """Create PolicyGptOssSafeguard instances for each section."""
        self.detectors: List[PolicyGptOssSafeguard] = []
        self.section_metadata: List[Dict[str, Any]] = []
        self.all_sections = filtered_sections
        self.section_id_to_index: Dict[str, int] = {}

        for idx, section in enumerate(filtered_sections):
            detector = PolicyGptOssSafeguard(
                policy_content=section["content"],
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                hub_name=hub_name,
                timeout=timeout if timeout is not None else 60,
                max_retries=max_retries if max_retries is not None else 3,
                api_key=api_key,
            )
            self.detectors.append(detector)
            section_id = section.get("section_id")
            self.section_metadata.append({
                "section_id": section_id,
                "header": section.get("metadata", {}).get("header") if isinstance(section.get("metadata"), dict) else None,
                "policy_id": policy_data.get("policy_id"),
            })
            if section_id:
                self.section_id_to_index[section_id] = idx
    
    def _populate_faiss_items(self, filtered_sections: List[Dict[str, Any]]) -> None:
        """Populate FAISS index with EmbeddingsItem objects."""
        try:
            # Build lookup dict for O(1) access
            section_lookup = {s.get("section_id"): s for s in filtered_sections}
            
            # Create EmbeddingsItem list matching FAISS index order
            faiss_items = []
            for faiss_idx_str in sorted(self.section_ids_mapping.keys(), key=int):
                section_id = self.section_ids_mapping[faiss_idx_str]
                section = section_lookup.get(section_id)
                if section:
                    section_content = section.get("content", "")
                    faiss_items.append(EmbeddingsItem(
                        text=section_content,
                        meta={"section_id": section_id, "faiss_index": faiss_idx_str}
                    ))
                else:
                    logger.warning(
                        "Section not found in filtered sections",
                        extra={"section_id": section_id, "faiss_index": faiss_idx_str}
                    )
            
            self.faiss_index.set_items(faiss_items)
            logger.info(
                "Populated FAISS index with items",
                extra={"item_count": len(faiss_items)}
            )
        except Exception as e:
            logger.warning(
                "Failed to populate FAISS index items",
                extra={"error": str(e), "section_count": len(filtered_sections)}
            )
            self.use_rag = False
    
    def __del__(self):
        """Cleanup resources."""
        # FAISS index doesn't need explicit cleanup
        # OpenAI client cleanup is handled by the client itself
        pass

    async def detect(self, query_string: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect policy violations by running section detectors in parallel batches.
        If RAG is enabled, retrieves only the most relevant sections first.

        Args:
            query_string: Content to classify

        Returns:
            DetectionResult with violation status and aggregated metadata
        """
        if len(self.detectors) == 0:
            return False, {"error": "No detectors configured"}

        # If RAG is enabled, retrieve relevant sections
        detectors_to_run = self.detectors
        metadata_to_run = self.section_metadata
        rag_info = None
        
        if self.use_rag and self.faiss_index is not None:
            try:
                # Retrieve top-k relevant sections
                results = await self.faiss_index.nearest_neighbor(
                    query_string, k=self.top_k, with_distance=True
                )
                
                # Filter by similarity threshold and map to section IDs
                relevant_section_ids = set()
                rag_scores = {}
                for item, similarity in results:
                    if similarity is not None and similarity >= self.similarity_threshold:
                        section_id = item.meta.get("section_id")
                        if section_id:
                            relevant_section_ids.add(section_id)
                            rag_scores[section_id] = similarity
                
                if relevant_section_ids:
                    # Filter detectors and metadata to only relevant sections
                    filtered_indices = []
                    for section_id in relevant_section_ids:
                        if section_id in self.section_id_to_index:
                            filtered_indices.append(self.section_id_to_index[section_id])
                    
                    # Sort by original order to maintain consistency
                    filtered_indices.sort()
                    detectors_to_run = [self.detectors[i] for i in filtered_indices]
                    metadata_to_run = [self.section_metadata[i] for i in filtered_indices]
                    
                    rag_info = {
                        "retrieved_sections": len(relevant_section_ids),
                        "total_sections": len(self.detectors),
                        "scores": rag_scores,
                    }
                    logger.info(
                        "RAG retrieved relevant sections",
                        extra={
                            "retrieved_count": len(relevant_section_ids),
                            "total_count": len(self.detectors),
                            "query_length": len(query_string),
                        }
                    )
                else:
                    logger.warning(
                        "RAG found no relevant sections above threshold",
                        extra={
                            "threshold": self.similarity_threshold,
                            "query_length": len(query_string),
                            "total_sections": len(self.detectors),
                        }
                    )
            except Exception as e:
                logger.warning(
                    "RAG retrieval failed",
                    extra={
                        "error": str(e),
                        "query_length": len(query_string),
                        "top_k": self.top_k,
                        "threshold": self.similarity_threshold,
                    }
                )
        
        # If max_parallel_sections is set, process in batches
        if self.max_parallel_sections and self.max_parallel_sections > 0:
            return await self._detect_batched(query_string, detectors_to_run, metadata_to_run, rag_info)
        else:
            return await self._detect_all_parallel(query_string, detectors_to_run, metadata_to_run, rag_info)

    async def _detect_all_parallel(
        self, 
        query_string: str,
        detectors: Optional[List[PolicyGptOssSafeguard]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        rag_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run detectors in parallel with fast fail."""
        if detectors is None:
            detectors = self.detectors
        if metadata is None:
            metadata = self.section_metadata
        
        # Create tasks with metadata
        task_data = [
            {
                "task": asyncio.create_task(detector.detect_with_time(query_string)),
                "section_idx": idx,
            }
            for idx, detector in enumerate(detectors)
        ]
        tasks = [td["task"] for td in task_data]

        violation_detected = False
        violation_result = None
        all_results = []

        while tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.cancelled():
                    continue
                try:
                    result = task.result()
                    # Find the section index for this task
                    section_idx = next(
                        td["section_idx"] for td in task_data if td["task"] == task
                    )
                    section_meta = metadata[section_idx]

                    all_results.append({
                        "section_id": section_meta["section_id"],
                        "header": section_meta["header"],
                        "hit": result.hit,
                        "result": result.result,
                        "exec_time": result.exec_time,
                    })

                    if result.hit:
                        violation_detected = True
                        violation_result = result
                        # Cancel remaining tasks and await cancellation
                        for p in pending:
                            p.cancel()
                        # Await cancelled tasks to avoid warnings
                        await asyncio.gather(*pending, return_exceptions=True)
                        break
                except Exception as e:
                    logger.warning(
                        "Detector task raised exception",
                        extra={
                            "error": str(e),
                            "query_length": len(query_string),
                            "section_idx": section_idx if 'section_idx' in locals() else None,
                        }
                    )
                    continue

            if violation_detected:
                break

            # Update task_data to only include pending tasks
            task_data = [td for td in task_data if td["task"] in pending]
            tasks = pending

        result_dict: Dict[str, Any] = {
            "violation": violation_detected,
            "sections": all_results,
        }
        
        if violation_detected and violation_result:
            result_dict.update({
                "violating_section": all_results[-1] if all_results else None,
                "model": violation_result.result.get("model"),
                "hub": violation_result.result.get("hub"),
                "response_string": self.blocked_response_string,
            })
        else:
            result_dict["response_string"] = query_string
        
        if rag_info:
            result_dict["rag_info"] = rag_info
        
        return violation_detected, result_dict

    async def _detect_batched(
        self,
        query_string: str,
        detectors: Optional[List[PolicyGptOssSafeguard]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        rag_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run detectors in batches with fast fail."""
        if detectors is None:
            detectors = self.detectors
        if metadata is None:
            metadata = self.section_metadata
        
        violation_detected = False
        all_results = []
        violation_result = None

        # Process in batches
        for batch_start in range(0, len(detectors), self.max_parallel_sections):
            batch_end = min(batch_start + self.max_parallel_sections, len(detectors))
            batch_detectors = detectors[batch_start:batch_end]
            batch_metadata = metadata[batch_start:batch_end]

            # Create tasks with metadata
            batch_task_data = [
                {
                    "task": asyncio.create_task(detector.detect_with_time(query_string)),
                    "task_idx": idx,
                }
                for idx, detector in enumerate(batch_detectors)
            ]
            tasks = [td["task"] for td in batch_task_data]

            batch_results = []
            batch_violation = False

            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    if task.cancelled():
                        continue
                    try:
                        result = task.result()
                        # Find the task index for this task
                        task_idx = next(
                            td["task_idx"] for td in batch_task_data if td["task"] == task
                        )
                        section_meta = batch_metadata[task_idx]

                        batch_results.append({
                            "section_id": section_meta["section_id"],
                            "header": section_meta["header"],
                            "hit": result.hit,
                            "result": result.result,
                            "exec_time": result.exec_time,
                        })

                        if result.hit:
                            batch_violation = True
                            violation_result = result
                            # Cancel remaining tasks in this batch and await cancellation
                            for p in pending:
                                p.cancel()
                            # Await cancelled tasks to avoid warnings
                            await asyncio.gather(*pending, return_exceptions=True)
                            break
                    except Exception as e:
                        logger.warning(
                            "Detector task raised exception",
                            extra={
                                "error": str(e),
                                "query_length": len(query_string),
                                "task_idx": task_idx if 'task_idx' in locals() else None,
                                "batch_start": batch_start if 'batch_start' in locals() else None,
                            }
                        )
                        continue

                if batch_violation:
                    break

                # Update batch_task_data to only include pending tasks
                batch_task_data = [td for td in batch_task_data if td["task"] in pending]
                tasks = pending

            all_results.extend(batch_results)

            if batch_violation:
                violation_detected = True
                break  # Fast fail - don't process remaining batches

        result_dict: Dict[str, Any] = {
            "violation": violation_detected,
            "sections": all_results,
        }
        
        if violation_detected and violation_result:
            result_dict.update({
                "violating_section": next(
                    (r for r in all_results if r["hit"]), None
                ),
                "model": violation_result.result.get("model"),
                "hub": violation_result.result.get("hub"),
                "response_string": self.blocked_response_string,
            })
        else:
            result_dict["response_string"] = query_string
        
        if rag_info:
            result_dict["rag_info"] = rag_info
        
        return violation_detected, result_dict
