#!/usr/bin/env bash
set -e  # exit immediately on error

# Convert CRLF line endings in all scripts
apt-get update && apt-get install -y dos2unix
find setup_scripts -type f -name "*.sh" -exec dos2unix {} +

# Continue with the rest of your setup
echo "$ROVERFLAKE_ROOT"

# check if VAR is unset or empty
if [ -z "${ROVERFLAKE_ROOT}" ]; then
  echo "ROVERFLAKE_ROOT is not set (or empty). That makes me really sad."
  sleep 2
  echo "Please set the environment variable in your bashrc to the root of RoverFlake2/"
  exit 1
fi
