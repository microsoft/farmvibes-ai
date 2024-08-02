# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List

from vibe_core.data import ExternalReference, ExternalReferenceList, gen_guid


def callback_builder():
    def callback(
        input_refs: List[ExternalReferenceList],
    ) -> Dict[str, List[ExternalReference]]:
        return {
            "ref_list": [
                ExternalReference.clone_from(refs, id=gen_guid(), url=url, assets=[])
                for refs in input_refs
                for url in refs.urls
            ]
        }

    return callback
