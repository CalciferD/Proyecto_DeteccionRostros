import streamlit as st
import cv2
import numpy as np
import os

# Título de la aplicación
st.title("Detector de Rostros")

def load_face_cascade():
    opencv_home = cv2.__file__
    folders = os.path.dirname(opencv_home)
    path = os.path.join(folders, 'data', 'haarcascade_frontalface_default.xml')
    if os.path.isfile(path):
        return cv2.CascadeClassifier(path)
    else:
        raise FileNotFoundError(f"Haar Cascade file not found at {path}")

# Función para detectar rostros en una imagen
def detect_faces(image):
    try:
        face_cascade = load_face_cascade()
        
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar rectángulos alrededor de los rostros detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return image, faces
    except Exception as e:
        st.error(f"Error en la detección de rostros: {str(e)}")
        return image, []

# Configurar la interfaz de usuario
def main():
    # Título principal
    st.title("Detector de Rostros con OpenCV")

    # Subtítulo
    st.header("Sube una imagen para detectar rostros:")

    # Widget para subir archivos
    uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        try:
            # Leer la imagen
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Mostrar la imagen subida
            st.image(image, channels="BGR", caption='Imagen original')

            # Botón para ejecutar la detección de rostros
            if st.button('Detectar Rostros'):
                # Ejecutar la detección de rostros
                image_with_faces, faces = detect_faces(image)

                # Crear columnas para mostrar la imagen con rostros detectados y el zoom en los rostros
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_with_faces, channels="BGR", caption='Imagen con rostros detectados')
                
                with col2:
                    for i, (x, y, w, h) in enumerate(faces):
                        # Recortar y mostrar cada rostro
                        face_crop = image[y:y+h, x:x+w]
                        st.image(face_crop, channels="BGR", caption=f'Rostro {i+1}', use_column_width=True)
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

if __name__ == '__main__':
    main()
