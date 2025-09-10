import { encodeWav } from "./encodeWav";
export async function mergeWavBlobs(blobs: Blob[], audioCtx: AudioContext): Promise<Blob> {
    // const audioCtx = new AudioContext();
    const validBlobs = blobs.filter(b => b.size > 0);
    const buffers = await Promise.all(
      validBlobs.map(blob =>
        blob.arrayBuffer().then(ab => audioCtx.decodeAudioData(ab))
      )
    );

    const totalLength = buffers.reduce((sum, b) => sum + b.length, 0);
    const numberOfChannels = buffers[0].numberOfChannels;
    const sampleRate = audioCtx.sampleRate;
    const output = audioCtx.createBuffer(numberOfChannels, totalLength, sampleRate);

    let offset = 0;
    for (const buffer of buffers) {
      for (let ch = 0; ch < numberOfChannels; ch++) {
        output.getChannelData(ch).set(buffer.getChannelData(ch), offset);
      }
      offset += buffer.length;
    }

    return encodeWav(output);
  }