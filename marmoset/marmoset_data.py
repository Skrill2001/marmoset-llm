import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, resumable_download, safe_extract

DATA_SPLIT = [
    "train",
    "val",
    "test",
]

def prepare_marmoset(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "all",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train', 'val', 'test'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "all" or dataset_parts[0] == "all":
        dataset_parts = DATA_SPLIT
    elif isinstance(dataset_parts, str):
        assert dataset_parts in DATA_SPLIT
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(dataset_parts=dataset_parts, output_dir=output_dir, prefix="marmoset")

    for part in tqdm(dataset_parts, desc="Preparing Marmoset Data parts"):
        if manifests_exist(part=part, output_dir=output_dir, prefix="marmoset"):
            logging.info(f"Marmoset data subset: {part} already prepared - skipping.")
            continue

        part_path = corpus_dir / part
        recordings = RecordingSet.from_dir(
            part_path,
            "*.wav",
            num_jobs=num_jobs,
            force_opus_sampling_rate=48000,
        )

        supervisions = []
        for recording in recordings:
            supervisions.append(
                SupervisionSegment(
                    id=recording.id,
                    recording_id=recording.id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    text="",
                )
            )

        supervisions = SupervisionSet.from_segments(supervisions)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        if output_dir is not None:
            supervisions.to_file(output_dir / f"marmoset_supervisions_{part}.jsonl.gz")
            recordings.to_file(output_dir / f"marmoset_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recordings, "supervisions": supervisions}
        print(f"Loaded {len(recordings)} recordings and {len(supervisions)} supervisions for '{part}'")

    return manifests


if __name__ == "__main__":

    prepare_marmoset(
        corpus_dir = "/cpfs02/user/housiyuan/dataset/monkey/codec_data",  
        dataset_parts = "all",
        output_dir = "/cpfs02/user/housiyuan/dataset/monkey/valle_data/manifests"
    )