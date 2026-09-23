# Data / release licensing note

This file is a release gate, not legal advice.

The previous local audit recorded a mismatch between the MangaSeg repository LICENSE/README language and license metadata embedded in the COCO-style dataset files. The repository-side terms were read as permitting academic/commercial use with attribution/citation, while COCO metadata carried non-commercial/share-alike wording.

Therefore:

- technical training and internal evaluation may proceed under the already acquired local datasets and their existing project handling;
- do not redistribute Manga109-s images or MangaSeg annotations from this repository;
- before commercial distribution of a trained model, preserve required citations/attribution and resolve the MangaSeg metadata-vs-LICENSE ambiguity with the dataset authors or appropriate legal review;
- this repository intentionally does not copy source datasets or license-controlled images.

Ultralytics was also not chosen as the project dependency because its current Free/Pro licensing is AGPL-based and proprietary closed-source product use may require a commercial/Enterprise license. The custom torch/torchvision code avoids adding that separate framework-license dependency.
