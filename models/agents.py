import numpy as np
from prisma.models import Cinema, Film, Genre, Screen

from models.utils import (Session, get_computed_spatial_map, get_probabilities,
                          get_probability_map, get_spatial_map, to_time)


class Agent:
    """An agent which generates schedules based on provided rules"""

    def __init__(
        self,
        cinema: Cinema,
        screens: list[Screen],
        genres: list[Genre],
        film_data: dict[int, tuple[Film, int, list[float]]]
    ) -> None:
        self.cinema = cinema
        self.screens = screens
        self.slots = int(24 * 60 / self.cinema.interval)
        self.genres = genres
        self.film_data = film_data
        self.state = None

        self.sessions: list[Session] = []
        self.reset()

    def calculate_state(self):
        """Calculates the state of the schedule

        Returns:
            NDArray: The matrix representing the schedule (screens, slots, films)
        """
        self.state = np.zeros(
            (len(self.screens), self.slots, len(self.film_data) + 1))

        for y, screen in enumerate(self.screens):
            screen_sessions = [d for d in self.sessions if d.y == y]
            spatial_map = get_spatial_map(self.cinema, screen_sessions)
            self.state[y, :, -1] = spatial_map

            for z, film_id in enumerate(self.film_data.keys()):
                film, _, film_dist = self.film_data[film_id]
                computed_map = get_computed_spatial_map(
                    spatial_map, self.cinema, film)
                dist = screen.capacity * film_dist
                prob_map = get_probability_map(dist, computed_map)
                self.state[y, :, z] = prob_map

        return self.state

    def reset(self):
        """Resets the state of the agent
        """
        self.sessions.clear()
        self.calculate_state()

    def done(self) -> bool:
        """Detects if the agent has made all possible moves

        Returns:
            bool: Indicates if the schedule is complete
        """
        return self.state[:, :, :-1].ravel().sum() == 0

    def take_action(self) -> Session:
        """Places a session based on the maximim probability

        Returns:
            Session: The placed session
        """
        probs = get_probabilities(self.state)
        idx = np.argmax(probs.probs)
        y, x, z = np.unravel_index(idx, shape=(
            len(self.screens), self.slots, len(self.film_data)))

        return self.take_determined_action(x, y, z)

    def take_determined_action(self, x, y, z):
        """Places a session at slot x, on screen y, for film z

        Args:
            x (int): The session start slot
            y (int): The screen index
            z (int): The film index

        Returns:
            Session: The placed session
        """
        film_id = list(self.film_data.keys())[z]
        film, slots, _ = self.film_data[film_id]
        session = Session(x=x, y=y, runtime=slots, film=film)
        self.sessions.append(session)

        return session

    def __str__(self):
        """Gets the string representation of the schedule

        Returns:
            str: The string representation of the schedule
        """
        result = ''
        for y, screen in enumerate(self.screens):
            result += f'{screen.name}, {screen.capacity}\n\n'

            sessions = [d for d in self.sessions if d.y == y]
            sessions.sort(key=lambda d: d.x)

            for session in sessions:
                start = to_time(session.x, self.cinema.interval)
                stop = to_time(session.x + session.runtime,
                               self.cinema.interval)
                result += f'{start} - {stop}\t{session.film.title}\n'

            result += '\n'

        return result
