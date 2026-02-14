#!/bin/bash

set -e

CONFIG_FILE="integration_private_keys.json"
ENV_FILE=".env"
CONFIG_URL="https://main.net955305.contentfabric.io/config"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed"
    exit 1
fi

# Read secrets from config file
VOD_SECRET=$(jq -r '.vod' "$CONFIG_FILE")
ASSETS_SECRET=$(jq -r '.assets' "$CONFIG_FILE")
LIVE_SECRET=$(jq -r '.live' "$CONFIG_FILE")

echo "Generating TEST_AUTH token..."
TEST_AUTH=$(qfab_cli content token create anonymous --config-url "$CONFIG_URL" --secret "$VOD_SECRET" | jq -r '.bearer')

echo "Generating ASSETS_AUTH token..."
ASSETS_AUTH=$(qfab_cli content token create anonymous --config-url "$CONFIG_URL" --secret "$ASSETS_SECRET" | jq -r '.bearer')

echo "Generating LIVE_AUTH token..."
LIVE_AUTH=$(qfab_cli content token create anonymous --config-url "$CONFIG_URL" --secret "$LIVE_SECRET" | jq -r '.bearer')

# Update or append to .env file
update_env_var() {
    local var_name=$1
    local var_value=$2

    if grep -q "^${var_name}=" "$ENV_FILE" 2>/dev/null; then
        # Update existing variable (macOS and Linux compatible)
        sed -i.bak "s|^${var_name}=.*|${var_name}=${var_value}|" "$ENV_FILE"
        rm -f "${ENV_FILE}.bak"
    else
        # Append new variable
        echo "${var_name}=${var_value}" >> "$ENV_FILE"
    fi
}

# Backup .env file
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "${ENV_FILE}.backup"
    echo "Backed up $ENV_FILE to ${ENV_FILE}.backup"
fi

# Update .env file
update_env_var "TEST_AUTH" "$TEST_AUTH"
update_env_var "ASSETS_AUTH" "$ASSETS_AUTH"
update_env_var "LIVE_AUTH" "$LIVE_AUTH"

echo "Successfully updated $ENV_FILE with auth tokens"
