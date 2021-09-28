from typing import Tuple, List

class Event:
    NOTE_ON = 'note_on'
    NOTE_OFF = 'note_off'
    TIME_SHIFT = 'time_shift'
    BAR = 'bar'
    CHORD = 'chord'

    def __init__(self, event_type: str, event_value: int = 0) -> None:
        self.event_type = event_type
        self.event_value = event_value

        if event_type == Event.NOTE_ON:
            while self.event_value < Encoding.MIN_NOTE:
                self.event_value += 12
            while self.event_value > Encoding.MAX_NOTE:
                self.event_value -= 12
        if event_type == Event.TIME_SHIFT:
            assert(event_value > 0 and event_value <= Encoding.MAX_SHIFT)
        if event_type == Event.CHORD:
            assert(event_value >= 0 and event_value <= Encoding.MAX_CHORD)
        assert(event_type in (Event.NOTE_ON, Event.NOTE_OFF,
                              Event.TIME_SHIFT, Event.BAR, Event.CHORD))

    def __repr__(self):
        return f"<Event {self.event_type} {self.event_value}>"


class Encoding:
    MIN_NOTE = 48
    MAX_NOTE = 84
    MAX_SHIFT = 16
    MAX_CHORD = 24
    _event_ranges: List[Tuple[str, int, int]]

    def __init__(self):
        self._event_ranges = [
            (Event.BAR, 0, 0),
            (Event.NOTE_OFF, 0, 0),
            (Event.TIME_SHIFT, 1, Encoding.MAX_SHIFT),
            (Event.NOTE_ON, Encoding.MIN_NOTE, Encoding.MAX_NOTE),
            (Event.CHORD, 0, Encoding.MAX_CHORD),
        ]

    @property
    def num_classes(self) -> int:
        return sum(max_value - min_value + 1
                   for _, min_value, max_value in self._event_ranges)

    def encode_event(self, event: Event) -> int:
        offset = 0
        for event_type, min_value, max_value in self._event_ranges:
            if event.event_type == event_type:
                return offset + event.event_value - min_value
            offset += max_value - min_value + 1

        raise ValueError('Unknown event type: %s' % event.event_type)

    def decode_event(self, index: int) -> Event:
        offset = 0
        for event_type, min_value, max_value in self._event_ranges:
            if offset <= index <= offset + max_value - min_value:
                return Event(
                    event_type=event_type,
                    event_value=min_value + index - offset,
                )
            offset += max_value - min_value + 1

        raise ValueError('Unknown event index: %s' % index)


