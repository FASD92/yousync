export function encodeWav(audioBuffer: AudioBuffer): Blob {
  console.log("")

    const numOfChan = audioBuffer.numberOfChannels,
          length = audioBuffer.length * numOfChan * 2 + 44,
          buffer = new ArrayBuffer(length),
          view = new DataView(buffer);
  
    const channels = [];
    let i, sample,
      offset = 0,
      pos = 0;
  
    setUint32(0x46464952); // "RIFF"
    setUint32(length - 8); // file length - 8
    setUint32(0x45564157); // "WAVE"
  
    setUint32(0x20746d66); // "fmt " chunk
    setUint32(16); // length = 16
    setUint16(1); // PCM (uncompressed)
    setUint16(numOfChan);
    setUint32(audioBuffer.sampleRate);
    setUint32(audioBuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
    setUint16(numOfChan * 2); // block-align
    setUint16(16); // 16-bit (hardcoded in this example)
  
    setUint32(0x61746164); // "data" - chunk
    setUint32(length - pos - 4); // chunk length
  
    for (i = 0; i < audioBuffer.numberOfChannels; i++)
      channels.push(audioBuffer.getChannelData(i));
  
    while (pos < length) {
      for (i = 0; i < numOfChan; i++) {
        sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF; // convert to 16-bit PCM
        view.setInt16(pos, sample, true); // little endian
        pos += 2;
      }
      offset++;
    }
  
    return new Blob([buffer], { type: "audio/wav" });
  
    function setUint16(data: number) {
      view.setUint16(pos, data, true);
      pos += 2;
    }
  
    function setUint32(data: number) {
      view.setUint32(pos, data, true);
      pos += 4;
    }
  }