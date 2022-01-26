from IPython.display import Audio

from IPython.core.display import display

sound_file = './sound/Bike-bell-sound.wav'

def ring():
	display(Audio(sound_file))