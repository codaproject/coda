"""Engine factories shared by language-support benchmarks."""


def make_whisper(size, language, device):
    import whisper

    model = whisper.load_model(size, device=device)

    def transcribe(path):
        return model.transcribe(
            str(path), language=language, fp16=(device != "cpu")
        )["text"]

    return transcribe


def make_faster_whisper(size, language, device, compute_type):
    from faster_whisper import WhisperModel

    model = WhisperModel(size, device=device, compute_type=compute_type)

    def transcribe(path):
        segments, _ = model.transcribe(str(path), language=language)
        return " ".join(segment.text for segment in segments)

    return transcribe
