#!/usr/bin/env node

const fs = require('fs');
const path = require("path")

/**
 * Converts normalized x-coordinates into a packed binary Buffer of 
 * 4-byte little-endian fixed-point integers (4 decimal places).
 * @param {Object} data - The JSON object structure provided.
 * @returns {Buffer} - The packed binary data.
 */
function packXCoordinates(data) {
    const xValues = [];

    // 1. Flatten all x-coordinates from the tags in order
    // Note: Assuming the tags are already sorted by frame_idx if order matters
    if (data.tags && Array.isArray(data.tags)) {
        data.tags.forEach(tag => {
            const coords = tag.additional_info?.['x-coordinates'];
            if (Array.isArray(coords)) {
                coords.forEach(x => {
                    // Convert to fixed point: 0.264327 -> 2643
                    const fixedPointValue = Math.round(x * 10000);
                    xValues.push(fixedPointValue);
                });
            }
        });
    }

    // 2. Allocate a buffer: 4 bytes per integer
    const buffer = Buffer.alloc(xValues.length * 4);

    // 3. Pack integers as 32-bit Little Endian (Int32LE)
    xValues.forEach((value, index) => {
        buffer.writeInt32LE(value, index * 4);
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
