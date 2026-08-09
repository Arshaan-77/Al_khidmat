from deepface import DeepFace
import numpy as np


# ==================================
# Face Validation
# ==================================

def validate_face(image_path):
    """
    Rules:
    1. Must contain exactly 1 face
    2. No face -> reject
    3. Multiple faces -> reject
    """

    try:

        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="opencv",
            enforce_detection=True
        )

        if len(faces) == 0:

            return (
                False,
                "No face detected in image."
            )

        if len(faces) > 1:

            return (
                False,
                "Multiple faces detected. Please upload a photo containing only one person."
            )

        return (
            True,
            "Valid face."
        )

    except Exception as e:

        return (
            False,
            f"Face validation failed: {str(e)}"
        )


# ==================================
# Generate Face Embedding
# ==================================

def generate_embedding(image_path):
    """
    Generate FaceNet512 embedding
    """

    try:

        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )

        embedding = result[0]["embedding"]

        return embedding

    except Exception as e:

        raise Exception(
            f"Embedding generation failed: {str(e)}"
        )


# ==================================
# Cosine Similarity
# ==================================

def cosine_similarity(
    embedding1,
    embedding2
):

    emb1 = np.array(
        embedding1,
        dtype=np.float32
    )

    emb2 = np.array(
        embedding2,
        dtype=np.float32
    )

    similarity = np.dot(
        emb1,
        emb2
    ) / (
        np.linalg.norm(emb1)
        * np.linalg.norm(emb2)
    )

    return float(similarity)


# ==================================
# Duplicate Face Check
# ==================================

def check_duplicate_face(
    new_embedding,
    existing_embeddings,
    threshold=0.80
):
    """
    existing_embeddings format:

    [
        {
            "beneficiary_id": 1,
            "embedding": [...]
        },
        {
            "beneficiary_id": 2,
            "embedding": [...]
        }
    ]
    """

    best_similarity = 0
    best_match = None

    for row in existing_embeddings:

        similarity = cosine_similarity(
            new_embedding,
            row["embedding"]
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_match = row["beneficiary_id"]

    if best_similarity >= threshold:

        return {
            "duplicate": True,
            "beneficiary_id": best_match,
            "similarity": round(
                best_similarity,
                4
            )
        }

    return {
        "duplicate": False,
        "beneficiary_id": None,
        "similarity": round(
            best_similarity,
            4
        )
    }
   

# ==================================
# Standalone Testing
# ==================================

if __name__ == "__main__":

    image_path = "test.jpg"

    valid, msg = validate_face(
        image_path
    )

    print(msg)

    if valid:

        embedding = generate_embedding(
            image_path
        )

        print(
            "Embedding Length:",
            len(embedding)
        )