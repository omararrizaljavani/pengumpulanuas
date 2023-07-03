# pengumpulan uas

from scipy import stats
import numpy as np

class FuzzyInferenceSystem:
    def __init__(self):
        self.speed = np.arange(0, 101, 1)
        self.pressure = np.arange(0, 101, 1)
        self.temperature = np.arange(0, 101, 1)

        # Fuzzy sets for Speed
        self.slow = self._membership(self.speed, [0, 0, 30, 50])
        self.steady = self._membership(self.speed, [40, 50, 60])
        self.fast = self._membership(self.speed, [50, 70, 100, 100])

        # Fuzzy sets for Pressure
        self.very_low = self._membership(self.pressure, [0, 0, 30, 50])
        self.low = self._membership(self.pressure, [40, 50, 60])
        self.medium = self._membership(self.pressure, [50, 70, 100, 100])
        self.high = self._membership(self.pressure, [80, 100, 100, 100])

        # Fuzzy sets for Temperature
        self.hot = self._membership(self.temperature, [50, 60, 100, 100])
        self.warm = self._membership(self.temperature, [30, 50, 70])
        self.cold = self._membership(self.temperature, [0, 0, 30, 50])
        self.freeze = self._membership(self.temperature, [0, 0, 10, 30])

    def _membership(self, x, params):
        return stats.trapzmf(x, params)

    def infer_temperature(self, speed_val, pressure_val):
        # Fuzzification
        speed_level = {
            'SLOW': self.slow(speed_val),
            'STEADY': self.steady(speed_val),
            'FAST': self.fast(speed_val)
        }
        pressure_level = {
            'VERY LOW': self.very_low(pressure_val),
            'LOW': self.low(pressure_val),
            'MEDIUM': self.medium(pressure_val),
            'HIGH': self.high(pressure_val)
        }

        # Rules
        rules = {
            'HOT': max(min(speed_level['SLOW'], pressure_level['VERY LOW']),
                       min(speed_level['STEADY'], pressure_level['VERY LOW']),
                       min(speed_level['FAST'], pressure_level['VERY LOW']),
                       min(speed_level['SLOW'], pressure_level['LOW'])),
            'WARM': max(min(speed_level['STEADY'], pressure_level['LOW']),
                        min(speed_level['FAST'], pressure_level['LOW']),
                        min(speed_level['SLOW'], pressure_level['MEDIUM']),
                        min(speed_level['STEADY'], pressure_level['MEDIUM'])),
            'COLD': max(min(speed_level['FAST'], pressure_level['MEDIUM']),
                        min(speed_level['SLOW'], pressure_level['HIGH']),
                        min(speed_level['STEADY'], pressure_level['HIGH']),
                        min(speed_level['FAST'], pressure_level['HIGH'])),
            'FREEZE': max(speed_level['SLOW'], pressure_level['VERY HIGH'],
                          speed_level['STEADY'], pressure_level['VERY HIGH'],
                          speed_level['FAST'], pressure_level['VERY HIGH'])
        }

        # Defuzzification (Centroid Method)
        num = 0
        den = 0
        for key, val in rules.items():
            num += self.temperature * val
            den += val

        return num / den

if __name__ == "__main__":
    fis = FuzzyInferenceSystem()

    # Input values
    speed_input = 25  # SLOW
    pressure_input = 15  # VERY LOW

    # Inference
    temperature_output = fis.infer_temperature(speed_input, pressure_input)
    print(f"Temperature Output: {temperature_output:.2f}")
