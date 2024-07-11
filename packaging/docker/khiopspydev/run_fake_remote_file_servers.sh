#!/bin/bash
# first start the docker daemon
service docker start

# File server for GCS
# WARNING : there are 3 major features actived by the options ...
# -v : the volume mounted exposes pre-provisioned files to be read remotely : the direct child folders will be the bucket names
# -filesystem-root : let upload and read new files remotely at the same location as the source
# -public-host : must expose localhost (https://github.com/fsouza/fake-gcs-server/issues/201)
docker run --rm \
  -d \
  -p 4443:4443 \
  --name fake-gcs-server \
  -v ./tests/resources/remote-access:/data fsouza/fake-gcs-server \
  -scheme http \
  -filesystem-root /data \
  -public-host localhost

# File server for S3
# WARNING :
# -v : the volume mounted exposes pre-provisioned files : the direct child folders will be the bucket names
#      these files were uploaded once because fake-s3 creates metadata
docker run --rm \
  -d \
  -p 4569:4569 \
  --name my_s3 \
  -v ./tests/resources/remote-access:/fakes3_root lphoward/fake-s3
