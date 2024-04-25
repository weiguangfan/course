import torch
import deepspeech
import pyaudio
import wave
import numpy as np
# 加载预训练的 DeepSpeech 模型
model = deepspeech.load_model('deepspeech-0.6.1-models/output_graph.pbmm')
model.eval()


# 实时音频流处理
def capture_audio_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    while True:
        data = stream.read(8000)
        yield data

# 语音识别
def recognize_speech(audio_data):
    # 对音频数据进行预处理（此处简化处理，实际可能需要更复杂的处理）
    audio_data = np.frombuffer(audio_data, dtype=np.int16)

    # 使用 DeepSpeech 进行语音识别
    text = model.stt(audio_data.tobytes(), 16000)
    return text

# 主函数
def main():
    audio_stream = capture_audio_stream()
    for data in audio_stream:
        recognized_text = recognize_speech(data)
        print(recognized_text)  # 实时输出识别结果

if __name__ == '__main__':
    main()
