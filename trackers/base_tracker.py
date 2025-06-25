from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def track(self, detections, frame=None):
        pass
