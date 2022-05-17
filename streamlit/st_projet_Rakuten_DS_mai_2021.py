import streamlit as st
import awesome_streamlit as ast

import pages.presentation.presentation
import pages.exploration.exploration_streamlit
import pages.classification_texte.texte_classification_et_prediction
import pages.classification_bimodale.classification_bimodale
import pages.prediction_bimodale.prediction_bimodale

# Load ast services
ast.core.services.other.set_logging_format()


# Dictionary of pages modules
PAGES = {
    "Présentation du projet": pages.presentation.presentation,
    "Exploration des données": pages.exploration.exploration_streamlit,
    "Développement et prédiction du modéle ML de Classification": pages.classification_texte.texte_classification_et_prediction,
    "Développement d'un modèle DL de Classification": pages.classification_bimodale.classification_bimodale,
    "Démo Prédiction DL": pages.prediction_bimodale.prediction_bimodale,
}


def main():
    """fonction principale de l'application"""

    # Création de la sidebar
    st.sidebar.title("Projet Rakuten")

    # mise en place de la sélection des pages
    selection = st.sidebar.radio("Menu", list(PAGES.keys()))
    page = PAGES[selection]

    # Chargement de la page sélectionnée
    with st.spinner("Chargement de {}...".format(selection)):
        ast.shared.components.write_page(page)

    def info(url):
        st.sidebar.markdown(
            f'<p style="background-color:#FFFF66;color:#0066ff;border-radius:2%;">{url}</p>',
            unsafe_allow_html=True,
        )

    def header(url):
        st.sidebar.markdown(
            f'<p style="color:#33ff33;">{url}</p>',
            unsafe_allow_html=True,
        )

    # Information sur l'équipe Projet
    info(
        "Réalisé par : <br />Haeji YUN, Mamadou LO<br />Christophe Paquet<br /><br />et notre coach Emilie Greff de Datascientest<br /><br />promotion continue DS mai 2021"
    )


if __name__ == "__main__":
    main()
