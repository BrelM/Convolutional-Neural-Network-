#include "Segmentation.h"

std::vector<cv::Mat> detectAndExtractFaces(const cv::Mat &image)
{
    std::vector<cv::Mat> faceSections;

    // Chargement du modèle pré-entraîné pour la détection de visages
    cv::CascadeClassifier faceCascade;
    faceCascade.load("path/to/your/haarcascade_frontalface_default.xml");

    // Détection des visages dans l'image
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(image, faces);

    // Parcours des visages détectés
    for (const auto &face : faces)
    {
        // Extraction de la section correspondant au visage
        cv::Mat faceSection = image(face);

        // Ajout de la section extraite au tableau
        faceSections.push_back(faceSection);

        // Dessin d'un rectangle autour du visage détecté
        cv::rectangle(image, face, cv::Scalar(0, 255, 0), 2);
    }

    // Affichage de l'image avec les rectangles entourant les visages détectés
    cv::imshow("Detection de visages", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return faceSections;
}
