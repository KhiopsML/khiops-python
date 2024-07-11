#!/bin/bash

ROOT_FOLDER=${1:-.} # defaults to current folder

# File server for GCS (runs in background)
# WARNING : there are 3 major features activated by the options ...
# -data : exposes pre-provisioned files (not currently used feature) : the direct child folder will be the bucket name
# -filesystem-root : let upload and read new files remotely at the same location as the source
# -public-host : must expose localhost (https://github.com/fsouza/fake-gcs-server/issues/201)
echo "Launching fake-gcs-server in background..."
nohup /bin/fake-gcs-server \
  -data "${ROOT_FOLDER}"/tests/resources/remote-access \
  -filesystem-root "${ROOT_FOLDER}"/tests/resources/remote-access \
  -scheme http \
  -public-host localhost > /dev/null < /dev/null 2>&1 & # needs to redirect all the 3 fds to free the TTY

# File server for S3 (runs in background)
# WARNING :
# -r : exposes pre-provisioned files (not currently used feature) : the direct child folder will be the bucket name
#      these files were uploaded once because fake-s3 creates metadata
echo "Launching fakes3 in background..."
PORT_NUMBER=${AWS_ENDPOINT_URL##*:}
nohup /usr/local/bin/fakes3 \
  -r "${ROOT_FOLDER}"/tests/resources/remote-access \
  -p "${PORT_NUMBER}" > /dev/null < /dev/null 2>&1 & # needs to redirect all the 3 fds to free the TTY
