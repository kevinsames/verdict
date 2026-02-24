#!/bin/bash
# Setup script for deploying Verdict to Azure Databricks
#
# Prerequisites:
# 1. Azure Databricks workspace
# 2. Databricks CLI installed (pip install databricks-cli)
# 3. Azure CLI installed (for Azure AD authentication)
# 4. Unity Catalog enabled in your workspace

set -e

echo "================================================"
echo "Verdict - Azure Databricks Setup"
echo "================================================"

# Check for required tools
echo ""
echo "Checking prerequisites..."

if ! command -v databricks &> /dev/null; then
    echo "❌ Databricks CLI not found. Install with: pip install databricks-cli"
    exit 1
fi

if ! command -v az &> /dev/null; then
    echo "⚠️  Azure CLI not found. Azure AD auth will not work."
    echo "   Install with: brew install azure-cli"
fi

echo "✓ Prerequisites check passed"

# Check for environment variables
echo ""
echo "Checking configuration..."

if [ -z "$DATABRICKS_HOST" ]; then
    echo "❌ DATABRICKS_HOST not set."
    echo "   Example: export DATABRICKS_HOST=https://adb-xxx.yyy.azuredatabricks.net"
    exit 1
fi

echo "✓ DATABRICKS_HOST: $DATABRICKS_HOST"

# Check authentication
if [ -n "$AZURE_TENANT_ID" ] && [ -n "$AZURE_CLIENT_ID" ] && [ -n "$AZURE_CLIENT_SECRET" ]; then
    echo "✓ Azure AD authentication configured"
elif [ -n "$DATABRICKS_TOKEN" ]; then
    echo "✓ PAT token configured"
else
    echo "⚠️  No authentication configured."
    echo "   Set either:"
    echo "   - Azure AD: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET"
    echo "   - PAT Token: DATABRICKS_TOKEN"
    exit 1
fi

# Build the wheel
echo ""
echo "Building verdict package..."
pip install build --quiet
python -m build --wheel
echo "✓ Package built"

# Validate bundle
echo ""
echo "Validating Databricks Asset Bundle..."
databricks bundle validate -t development
echo "✓ Bundle validated"

# Deploy bundle
echo ""
echo "Deploying to Azure Databricks..."
databricks bundle deploy -t development
echo "✓ Bundle deployed"

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Create prompt dataset (run notebooks/create_sample_dataset.py)"
echo "2. Deploy your model to Model Serving endpoint"
echo "3. Run the pipeline:"
echo "   databricks bundle run verdict_pipeline -t development"
echo ""