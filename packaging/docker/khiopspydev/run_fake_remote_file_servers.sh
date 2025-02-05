#!/bin/bash

ROOT_FOLDER=${1:-.} # defaults to current folder

# File server for S3 (runs in background)
# WARNING :
# -r : exposes pre-provisioned files (not currently used feature) : the direct child folder will be the bucket name
#      these files were uploaded once because fake-s3 creates metadata
echo "Launching fakes3 in background..."
PORT_NUMBER=${AWS_ENDPOINT_URL##*:}
nohup /usr/local/bin/fakes3 \
  -r "${ROOT_FOLDER}"/tests/resources/remote-access \
  -p "${PORT_NUMBER}" > /dev/null < /dev/null 2>&1 & # needs to redirect all the 3 fds to free the TTY
