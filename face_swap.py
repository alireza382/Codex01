"""Example offline face swap using insightface.

The resulting image is synthetic. Use responsibly and clearly label any
generated content. Do not mislead or harm others with this script.

Usage:
    python face_swap.py src.jpg dst.jpg output.jpg
"""

import sys
import cv2
import insightface


def main(src_path: str, dst_path: str, output_path: str) -> None:
    """Swap the primary face from src_path onto dst_path and save to output_path."""
    # Initialize face analysis for detection and alignment
    app = insightface.app.FaceAnalysis(name="antelopev2")
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load the face swap model
    swapper = insightface.model_zoo.get_model("inswapper_128.onnx")

    img_src = cv2.imread(src_path)
    img_dst = cv2.imread(dst_path)

    src_faces = app.get(img_src)
    dst_faces = app.get(img_dst)

    if not src_faces:
        raise RuntimeError(f"No face detected in source image {src_path}")
    if not dst_faces:
        raise RuntimeError(f"No face detected in destination image {dst_path}")

    src_face = src_faces[0]
    dst_face = dst_faces[0]

    swapped = swapper.get(img_dst, dst_face, src_face, paste_back=True)
    cv2.imwrite(output_path, swapped)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])

