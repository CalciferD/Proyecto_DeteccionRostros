import streamlit as st
import cv2
import numpy as np
import face_recognition

# Título de la aplicación
st.title("Detector de Rostros")

# Función para detectar rostros en una imagen
def detect_faces(image):
    try:
        # Convertir de BGR a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        # Dibujar rectángulos alrededor de los rostros detectados
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

        return image, face_locations
    except Exception as e:
        st.error(f"Error en la detección de rostros: {str(e)}")
        return image, []

# Configurar la interfaz de usuario
def main():
    # Título principal
    st.title("Detector de Rostros Avanzado")

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
                image_with_faces, face_locations = detect_faces(image)

                # Crear columnas para mostrar la imagen con rostros detectados y el zoom en los rostros
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_with_faces, channels="BGR", caption='Imagen con rostros detectados')
                
                with col2:
                    for i, (top, right, bottom, left) in enumerate(face_locations):
                        # Recortar y mostrar cada rostro
                        face_crop = image[top:bottom, left:right]
                        st.image(face_crop, channels="BGR", caption=f'Rostro {i+1}', use_column_width=True)

        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")

if __name__ == '__main__':
    main()
