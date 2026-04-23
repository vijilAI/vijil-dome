#!/bin/bash
# One-time setup: Create OIDC provider + IAM role for GitHub Actions
# to access the vijil-inference S3 bucket (read-only, model downloads).
#
# Run once from an authenticated AWS session:
#   aws sso login && bash infra/ci-s3-access/create-oidc-role.sh
#
# Prerequisites: aws CLI authenticated with IAM admin access

set -euo pipefail

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-west-2"
ROLE_NAME="github-actions-dome-ci"
BUCKET="vijil-inference"
GITHUB_ORG="vijilAI"
GITHUB_REPO="vijil-dome"

echo "Account: $ACCOUNT_ID"
echo "Role:    $ROLE_NAME"
echo "Bucket:  $BUCKET"
echo "Repo:    $GITHUB_ORG/$GITHUB_REPO"
echo ""

# Step 1: Create OIDC provider for GitHub Actions (idempotent)
echo "=== Step 1: OIDC Provider ==="
OIDC_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN" 2>/dev/null; then
    echo "OIDC provider already exists"
else
    # GitHub's OIDC thumbprint (stable, well-known)
    aws iam create-open-id-connect-provider \
        --url "https://token.actions.githubusercontent.com" \
        --client-id-list "sts.amazonaws.com" \
        --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1"
    echo "Created OIDC provider"
fi

# Step 2: Create IAM role with trust policy scoped to this repo
echo ""
echo "=== Step 2: IAM Role ==="
TRUST_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
            },
            "Action": "sts:AssumeRoleWithWebIdentity",
            "Condition": {
                "StringEquals": {
                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
                },
                "StringLike": {
                    "token.actions.githubusercontent.com:sub": "repo:${GITHUB_ORG}/${GITHUB_REPO}:*"
                }
            }
        }
    ]
}
EOF
)

if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
    echo "Role already exists — updating trust policy"
    aws iam update-assume-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-document "$TRUST_POLICY"
else
    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --description "Read-only S3 access for vijil-dome CI model downloads"
    echo "Created role: $ROLE_NAME"
fi

# Step 3: Attach S3 read-only policy scoped to the inference bucket
echo ""
echo "=== Step 3: S3 Policy ==="
S3_POLICY=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${BUCKET}",
                "arn:aws:s3:::${BUCKET}/models/vijil/*"
            ]
        }
    ]
}
EOF
)

POLICY_NAME="dome-ci-s3-read"
aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "$POLICY_NAME" \
    --policy-document "$S3_POLICY"
echo "Attached inline policy: $POLICY_NAME"

echo ""
echo "=== Done ==="
echo "Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo ""
echo "Add this to the GitHub Actions workflow:"
echo "  role-to-assume: arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
