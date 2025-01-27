#!/bin/bash

export VIDEO_AUTH=$(qfab_cli content token create iq__3C58dDYxsn5KKSWGYrfYr44ykJRm --config ../configs/prod-mgm.json --update | jq -r '.bearer')

export ASSETS_AUTH=$(qfab_cli content token create iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2 --config ../configs/prod-eluvio.json --update | jq -r '.bearer')

export VIDEO_WRITE=$(qfab_cli content edit iq__3C58dDYxsn5KKSWGYrfYr44ykJRm --config ../configs/prod-mgm.json | jq -r '.q.write_token')

export ASSETS_WRITE=$(qfab_cli content edit iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2 --config ../configs/prod-eluvio.json | jq -r '.q.write_token')