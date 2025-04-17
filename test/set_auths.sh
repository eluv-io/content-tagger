#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AUTH_iq__3C58dDYxsn5KKSWGYrfYr44ykJRm=$(qfab_cli content token create iq__3C58dDYxsn5KKSWGYrfYr44ykJRm --config $DIR/../configs/prod-mgm.json --update | jq -r '.bearer')

export AUTH_iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2=$(qfab_cli content token create iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2 --config $DIR/../configs/prod-eluvio.json --update | jq -r '.bearer')

export AUTH_hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU=$(qfab_cli content token create hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU --config $DIR/../configs/ml03-mgm.json --update | jq -r '.bearer')

export WRITE_iq__3C58dDYxsn5KKSWGYrfYr44ykJRm=$(qfab_cli content edit iq__3C58dDYxsn5KKSWGYrfYr44ykJRm --config $DIR/../configs/prod-mgm.json | jq -r '.q.write_token')

export WRITE_iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2=$(qfab_cli content edit iq__4BT8BBNEEDvysXqjZgj4BRA5jVo2 --config $DIR/../configs/prod-eluvio.json | jq -r '.q.write_token')

export WRITE_hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU=$(qfab_cli content edit hq__3B47zhoJbyiwqWUq8DNJJQXHg1GZitfQBXpsGkV2tQLpHzp2McAk7xAFJwKSJ99mgjzZjqRdHU --config $DIR/../configs/prod-mgm.json | jq -r '.q.write_token')