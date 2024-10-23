#!/bin/bash

# We cannot start the fake remote files servers until the source code is fetched
# because we expose pre-provisioned files to be read

ROOT_FOLDER=${1:-.} # defaults to current folder

# File server for GCS (runs in background)
# WARNING : there are 3 major features actived by the options ...
# -data : exposes pre-provisioned files to be read remotely : the direct child folders will be the bucket names
# -filesystem-root : let upload and read new files remotely at the same location as the source
# -public-host : must expose localhost (https://github.com/fsouza/fake-gcs-server/issues/201)
nohup /bin/fake-gcs-server \
  -data "${ROOT_FOLDER}"/tests/resources/remote-access \
  -filesystem-root "${ROOT_FOLDER}"/tests/resources/remote-access \
  -scheme http \
  -public-host localhost &

# File server for S3 (runs in background)
# WARNING :
# -r : exposes pre-provisioned files : the direct child folders will be the bucket names
#      these files were uploaded once because fake-s3 creates metadata
PORT_NUMBER=${AWS_ENDPOINT_URL##*:}
nohup /usr/local/bin/fakes3 \
  -r "${ROOT_FOLDER}"/tests/resources/remote-access \
  -p "${PORT_NUMBER:-4569}" &
