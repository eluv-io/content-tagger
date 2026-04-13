#!/usr/bin/env node

const fs = require('fs');
const path = require("path")
const { ElvClient } = require("@eluvio/elv-client-js");

async function makeFabricClient() {
  const network = process.env.FABRIC_NETWORK || "main"
  console.debug(`makeFabricClient -- network ${network}`)
  
  const client = await ElvClient.FromNetworkName({networkName: network})
  const wallet = client.GenerateWallet();
  const signer = wallet.AddAccount({
    privateKey: process.env.PRIVATE_KEY,
  });
  client.SetSigner({ signer });
  return client
}

const TAGSTORE_BASE = process.env.TAGSTORE_URL || "https://ai.contentfabric.io/tagstore"

async function readTagstoreData(client, contentId) {
  const tagData = {}
  
  const scToken = await client.GenerateStateChannelToken({ objectId: contentId }).catch((err) => { console.error(contentId, err) })
  
  if (scToken) {
    const params = new URLSearchParams({
      track: "vertical_video",
      limit: 10000,
      has_frame_info: true,
    })
    
    const response = await fetch(`${TAGSTORE_BASE}/${contentId}/tags?${params}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${scToken}`
      }
    })
    
    if (response.status !== 200) {
      throw new Error(`Error fetching tags for ${contentId}: ${response.status} ${response.statusText} ${await response.text()}`)
    }
    
    return await response.json()
  }

  return { "tags": {} }
}

async function writeBinaryFile(client, iq, bufferData) {

  const libraryId = await client.ContentObjectLibraryId({objectId: iq});
  
  const editResponse = await client.EditContentObject({
    libraryId,
    objectId: iq
  })
                                                     
  console.debug(`iq:${iq} write_token:${editResponse.write_token}`)  
  await client.UploadFiles({
    libraryId,
    objectId: iq,
    writeToken: editResponse.write_token,
    fileInfo: [
      {
        path: `vertical.bin`,
        mime_type: "application/octet-stream",
        size: bufferData.length,
        data: bufferData,
      },
    ],
  })

  const resp = await client.FinalizeContentObject({
    libraryId,
    objectId: iq,
    writeToken: editResponse.write_token,
    commitMessage: process.env.COMMIT_MESSAGE || `${path.basename(__filename)}: Upload vertical video information file (${process.env.USER})`,
  });
  console.log(`iq::${iq} UPLOAD COMPLETE hash:${resp.hash}`)

}

/**
 * Sorts tags, validates frame continuity, and packs x-coordinates into a binary Buffer.
 * @param {Object} data 
 * @returns {Buffer}
 */
function packXCoordinates(data) {
    if (!data.tags || !Array.isArray(data.tags)) {
        return Buffer.alloc(0);
    }

    // 1. Sort the tags by frame_idx to ensure chronological processing
    const sortedTags = [...data.tags].sort((a, b) => {
        return (a.frame_info?.frame_idx || 0) - (b.frame_info?.frame_idx || 0);
    });

    const xValues = [];
    let expectedNextFrame = sortedTags.length > 0 ? sortedTags[0].frame_info.frame_idx : 0;

    let breaks = 0
    // 2. Iterate and validate
    sortedTags.forEach((tag, index) => {
        const currentFrameIdx = tag.frame_info?.frame_idx ?? 0;
        const coords = tag.additional_info?.['x-coordinates'] || [];

        // Emit warning if there is a gap or overlap in frame indices
        if (currentFrameIdx !== expectedNextFrame) {
            breaks++
            if (breaks < 4) console.warn(
                `[Warning] Continuity break at Tag ID: ${tag.id}. ` +
                `Expected frame_idx ${expectedNextFrame}, but found ${currentFrameIdx}.`
            );
        }

        // Add coordinates to our list
        coords.forEach(x => {
            // Fixed point with 4 decimal places
            xValues.push(Math.round(x * 10000));
        });

        // Calculate what the next frame_idx should be
        // (Current index + number of samples provided in this tag)
        expectedNextFrame = currentFrameIdx + coords.length;
    });

    // 3. Pack into Buffer as 4-byte Little Endian integers
    const buffer = Buffer.alloc(xValues.length * 4);
    xValues.forEach((value, i) => {
        buffer.writeInt32LE(value, i * 4);
    });

    if (breaks > 0) {
      console.error(`There were ${breaks} breaks in continuity. Err`)
      return null
    }
    return buffer;
}

async function main(iqs) {
  const client = await makeFabricClient()
  
  for (const iq of iqs) {
    console.log(`------------- ${iq} -------------`)
    await processVV(client, iq)
  }
}
  
async function processVV(client, inputIq) {
  try {        
    const jsonData = await readTagstoreData(client, inputIq)

    if (jsonData.tags.length < 1) {
      console.log(`${inputIq}:: no vertical data, not writing to fabric`)
    }
    
    const packedBuffer = packXCoordinates(jsonData);
    if (packedBuffer == null) console.error(`${inputIq}:: failed to pack`)

    await writeBinaryFile(client, inputIq, packedBuffer);
    
  } catch (err) {
    console.error("Error processing files:", err.message);
  }
}

if (require.main === module) {

  let args = process.argv
  
  while (args.length) {
    let arg = args.shift()
    if (arg.endsWith(path.basename(__filename))) break
  }

  main(args)
}
