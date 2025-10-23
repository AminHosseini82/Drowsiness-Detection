import simpleaudio as sa
wave = sa.WaveObject.from_wave_file(r"F:\University\7th term\Computer vision\project\Drowsiness-Detection\alarms\alarm_high.wav")
play = wave.play()
play.wait_done()
print("sound done!")
