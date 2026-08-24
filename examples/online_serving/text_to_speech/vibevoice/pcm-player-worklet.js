// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

class PCMPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Thirty seconds at the AudioContext rate. Playback drains concurrently;
    // overflow is reported instead of silently crossing request buffers.
    this.capacity = Math.max(1, Math.floor(sampleRate * 30));
    this.ring = new Float32Array(this.capacity);
    this.readIndex = 0;
    this.writeIndex = 0;
    this.available = 0;
    this.underrunFrames = 0;
    this.overflowSamples = 0;
    this.processCalls = 0;
    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "reset") {
        this.readIndex = 0;
        this.writeIndex = 0;
        this.available = 0;
        this.underrunFrames = 0;
        this.overflowSamples = 0;
        this.postBufferState();
        return;
      }
      if (message.type !== "samples" || !message.samples) {
        return;
      }
      const samples = new Float32Array(message.samples);
      for (let index = 0; index < samples.length; index += 1) {
        if (this.available === this.capacity) {
          this.overflowSamples += samples.length - index;
          break;
        }
        this.ring[this.writeIndex] = samples[index];
        this.writeIndex = (this.writeIndex + 1) % this.capacity;
        this.available += 1;
      }
      this.postBufferState();
    };
    this.postBufferState();
  }

  postBufferState() {
    this.port.postMessage({
      type: "buffer",
      available: this.available,
      capacity: this.capacity,
      overflowSamples: this.overflowSamples,
    });
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) {
      return true;
    }
    const channel = output[0];
    let underrun = false;
    for (let index = 0; index < channel.length; index += 1) {
      if (this.available > 0) {
        channel[index] = this.ring[this.readIndex];
        this.readIndex = (this.readIndex + 1) % this.capacity;
        this.available -= 1;
      } else {
        channel[index] = 0;
        underrun = true;
      }
    }
    for (let channelIndex = 1; channelIndex < output.length; channelIndex += 1) {
      output[channelIndex].set(channel);
    }
    if (underrun) {
      this.underrunFrames += 1;
      if (this.underrunFrames % 100 === 1) {
        this.port.postMessage({
          type: "underrun",
          frames: this.underrunFrames,
        });
      }
    }
    this.processCalls += 1;
    if (this.processCalls % 25 === 0) this.postBufferState();
    return true;
  }
}

registerProcessor("pcm-player", PCMPlayerProcessor);
