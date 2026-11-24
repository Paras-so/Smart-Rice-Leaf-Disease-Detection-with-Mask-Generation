# Disease-region annotation pilot

Open `../../pixel_level_disease_annotation.ipynb`. Its saved outputs show six original images with binary masks and red overlays. The six drafts cover 3 training, 1 validation, and 2 test images. A healthy image has an empty mask. The other five examples cover Bacterial Leaf Blight, Brown Spot, Leaf Blast, Rice Hispa, and Sheath Blight. The pilot does not cover all nine classes.

These are assistant-traced draft regions. Hispa additionally uses pale-pixel selection within a traced region. They require user boundary/missed-region review and are not expert ground truth. Initial masks include contiguous discolored margins; review that policy before scaling.

Run the editor cell in the notebook. Click polygon vertices, close each region, choose a 1?5 rating and a review decision, then save. Ratings and reviewed outputs are separate from drafts. No acceptance or rating is recorded by the assistant.

PNG label values are 0 = unmarked and 1 = visibly affected tissue. They match EXIF-transposed source resolution. White preview masks scale label 1 to 255. Image coverage percentages are not disease severity because the denominator includes background.

The user authorized full-dataset generation on 15 November 2026. See `../pixel_annotations_full/` and the full-run sections of the notebook. Full authorization does not fill in ratings or accept each draft mask. Automatic proposals and geometrically matched augmented masks are generated separately from these pilot drafts and still require individual review.

Validation performed: executed notebook (interactive cell excluded); native mask dimensions, labels, and pixel counts; healthy empty mask; six preview/mask API endpoints; mask/rating save in an isolated temporary directory; invalid-rating rejection; full-dataset gate. Browser preview was unavailable, so live UI rendering was not verified.
