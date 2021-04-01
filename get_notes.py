import pretty_midi
midi_data = pretty_midi.PrettyMIDI('C:/Users/thoma/Downloads/BanjoKazooie - Pause Menu.mid')

print("duration:",midi_data.get_end_time())
print(f'{"note":>10} {"start":>10} {"end":>10}')
for instrument in midi_data.instruments:
    print("instrument:", instrument.program);
    for note in instrument.notes:
        print(f'{note.pitch:10} {note.start:10} {note.end:10}')
		print('hey')