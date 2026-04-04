#!/usr/bin/env node

const fs = require('fs');
const path = require("path")

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

    // 2. Iterate and validate
    sortedTags.forEach((tag, index) => {
        const currentFrameIdx = tag.frame_info?.frame_idx ?? 0;
        const coords = tag.additional_info?.['x-coordinates'] || [];

        // Emit warning if there is a gap or overlap in frame indices
        if (currentFrameIdx !== expectedNextFrame) {
            console.warn(
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

    return buffer;
}

/**
 * Main driver to read JSON, process, and write binary output.
 * @param {string} inputPath 
 * @param {string} outputPath 
 */
function main(inputPath, outputPath) {
    try {
        // Read and parse the JSON file
        const rawData = fs.readFileSync(inputPath, 'utf8');
        const jsonData = JSON.parse(rawData);

        // Process the data
        const packedBuffer = packXCoordinates(jsonData);

        // Write the binary file
        fs.writeFileSync(outputPath, packedBuffer);

        console.log(`Successfully packed ${packedBuffer.length / 4} integers into ${outputPath}`);
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
  
  main(args[0], args[1])
}
