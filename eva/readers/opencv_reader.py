# coding=utf-8
# Copyright 2018-2022 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Iterator

import cv2
import numpy as np
from decord import AVReader
from eva.expression.abstract_expression import AbstractExpression
from eva.expression.expression_utils import extract_range_list_from_predicate
from eva.readers.abstract_reader import AbstractReader
from eva.utils.logging_manager import logger


class OpenCVReader(AbstractReader):
    def __init__(
        self,
        *args,
        predicate: AbstractExpression = None,
        sampling_rate: int = None,
        read_audio: bool = False,
        read_video: bool = True,
        **kwargs
    ):
        """Read frames from the disk

        Args:
            predicate (AbstractExpression, optional): If only subset of frames
            need to be read. The predicate should be only on single column and
            can be converted to ranges. Defaults to None.
            sampling_rate (int, optional): Set if the caller wants one frame
            every `sampling_rate` number of frames. For example, if `sampling_rate = 10`, it returns every 10th frame. If both `predicate` and `sampling_rate` are specified, `sampling_rate` is given precedence.
        """
        self._predicate = predicate
        self._sampling_rate = sampling_rate or 1
        self._read_audio = read_audio
        self._read_video = read_video
        super().__init__(*args, **kwargs)

    def _read(self) -> Iterator[Dict]:
        video = cv2.VideoCapture(self.file_url) if self._read_video else None
        av = (
            AVReader(self.file_url, mono=True, sample_rate=16000)
            if self._read_audio
            else None
        )
        num_frames = (
            int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if self._read_video else len(av)
        )
        if self._predicate:
            range_list = extract_range_list_from_predicate(
                self._predicate, 0, num_frames - 1
            )
        else:
            range_list = [(0, num_frames - 1)]
        logger.debug("Reading frames")
        if self._sampling_rate == 1:
            video_frame = np.empty((0, 0, 0))
            audio_frame = np.empty(0)
            for begin, end in range_list:
                frame_id = begin
                if self._read_video:
                    video.set(cv2.CAP_PROP_POS_FRAMES, begin)
                    _, video_frame = video.read()
                if self._read_audio:
                    audio_frame = av[frame_id][0].asnumpy()
                while video_frame is not None and frame_id <= end:
                    if self._read_audio:
                        audio_frame = av[frame_id][0].asnumpy()
                    yield {
                        "id": frame_id,
                        "data": video_frame,
                        "seconds": frame_id // video.get(cv2.CAP_PROP_FPS)
                        if self._read_video
                        else 0,
                        "audio": audio_frame,
                    }
                    if self._read_video:
                        _, video_frame = video.read()
                    frame_id += 1
        else:
            for begin, end in range_list:
                # align begin with sampling rate
                if begin % self._sampling_rate:
                    begin += self._sampling_rate - (begin % self._sampling_rate)
                for frame_id in range(begin, end + 1, self._sampling_rate):
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    _, frame = video.read()
                    if frame is not None:
                        yield {"id": frame_id, "data": frame}
                    else:
                        break
