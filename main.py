import os
import cv2
import face_recognition

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_images_from_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def save_face_image(image, face_location, output_path):
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    cv2.imwrite(output_path, face_image_rgb)

def organize_images(image_folder, output_folder, tolerance=0.6):
    ensure_dir(output_folder)

    known_faces = []
    face_folders = []

    image_files = get_images_from_folder(image_folder)

    for image_file in image_files:
        image = face_recognition.load_image_file(image_file)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for face_location, face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance)

            if True in matches:
                match_index = matches.index(True)
                person_folder = face_folders[match_index]
            else:
                known_faces.append(face_encoding)
                person_folder = os.path.join(output_folder, f"person_{len(known_faces)}")
                face_folders.append(person_folder)
                ensure_dir(person_folder)

            # Save the face image of the person
            face_image_path = os.path.join(person_folder, "face.jpg")
            save_face_image(image, face_location, face_image_path)

            # Draw rectangle around face
            # top, right, bottom, left = face_location
            # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Convert the image from RGB to BGR format (OpenCV uses BGR format)
            annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Save the annotated image to the appropriate folder
            output_image_path = os.path.join(person_folder, os.path.basename(image_file))
            cv2.imwrite(output_image_path, annotated_image)

if __name__ == "__main__":
    image_folder = "./test"
    output_folder = "./people"

    organize_images(image_folder, output_folder, tolerance=0.4)
