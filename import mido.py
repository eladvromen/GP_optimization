import mido 
from mido import MidiFile

# load the MIDI file
midi_file = MidiFile("/Users/eladv/OneDrive/מסמכים/research/Bohemian-Rhapsody-1.mid")

# create an empty list to store the right hand notes
right_hand_notes = []



# iterate over the tracks in the MIDI file
for track in midi_file.tracks:
    # iterate over the messages in the track
    midi.mess
    for message in track:
        # check if the message is a note on message
        if message.type == 'note_on':
            # check if the message is from the right hand channel
            if message.channel == 0:
                # add the note to the right hand notes list
                right_hand_notes.append(message.note)

# print the right hand notes
print(right_hand_notes)

midi_file.save('C:/Users/eladv/OneDrive/מסמכים/research/right_hand_bohemian.mid')
