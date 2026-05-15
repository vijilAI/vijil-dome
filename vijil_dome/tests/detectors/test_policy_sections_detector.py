"""BC-8: PolicySectionsDetector concurrency default and error tracking."""

from __future__ import annotations

import inspect


class TestPolicySectionsDefaults:
    """PolicySectionsDetector should have a bounded default concurrency."""

    def test_default_max_parallel_sections(self) -> None:
        from vijil_dome.detectors.methods.policy_sections_detector import (
            PolicySectionsDetector,
        )

        sig = inspect.signature(PolicySectionsDetector.__init__)
        default = sig.parameters["max_parallel_sections"].default
        assert default is not None
        assert isinstance(default, int)
        assert default <= 20
