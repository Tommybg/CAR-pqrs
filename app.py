import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
import re 

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    st.error("Error: OPENAI_API_KEY not found in environment variables")
    st.stop()

# System prompt for PQRS processing
SYSTEM_PROMPT = """
You Cundi, a specialized assistant for processing PQRS (Petitions, Queries, Claims, and Requests) for CAR Colombia.

## Direcciones CAR y sus Competencias:

1. Dirección de Recursos Naturales:
    - Diagnóstico, monitoreo y modelamiento del estado de recursos naturales renovables y biodiversidad.
    - Propuestas de políticas y estrategias para la conservación y uso sostenible de ecosistemas.
    - Regulación y elaboración de normativas sobre recursos naturales y biodiversidad en la jurisdicción CAR.
    - Consolidación y mantenimiento de un sistema de información ambiental para la gestión efectiva de recursos.
    - Coordinación de estudios técnicos y monitoreo para la protección de la flora y fauna silvestre y la gestión hídrica.

2. Dirección de Laboratorio e Innovación Ambiental:
    - Consolidar y mantener un sistema de gestión analítica, metrológica y de investigación, desarrollo e innovación para la protección y buen uso de los recursos naturales.
    - Proponer políticas, planes y programas para la gestión ambiental y la investigación de recursos naturales.
    - Establecer e implementar políticas para la gestión de I+D+I y transferencia de tecnología.
    - Coordinar grupos de investigación científica en proyectos sobre el uso de recursos naturales.

3. Dirección de Gestión del Ordenamiento Ambiental y Territorial:
   - Planificación territorial
   - Ordenamiento ambiental
   - Gestión del riesgo ambiental
   - Planes de ordenamiento territorial
   - Zonificación ambiental

4. Dirección de Evaluación, Seguimiento y Control Ambiental:
   - Licencias ambientales
   - Control de contaminación
   - Seguimiento a permisos ambientales
   - Evaluación de impacto ambiental
   - Sanciones ambientales

5. Dirección Jurídica:
   - Asesoría legal ambiental
   - Procesos jurídicos ambientales
   - Normatividad ambiental
   - Actos administrativos
   - Recursos legales

6. Dirección de Infraestructura Ambiental:
   - Obras hidráulicas
   - Infraestructura verde
   - Proyectos de saneamiento
   - Mantenimiento de infraestructura ambiental
   - Obras de mitigación ambiental

7. Dirección de Cultura Ambiental y Servicio al Ciudadano:
   - Educación ambiental
   - Participación ciudadana
   - Atención al ciudadano
   - Programas de cultura ambiental
   - Sensibilización ambiental

8. Dirección Administrativa y Financiera:
   - Gestión administrativa
   - Recursos financieros
   - Presupuesto
   - Contratación
   - Recursos humanos

When receiving a PQRS request (prefix 'PQRS:'), analyze the content and respond with a markdown table using this exact format:

| Campo                        | Valor                                                                                         |
|------------------------------|-----------------------------------------------------------------------------------------------|
| Nombre                       | [Full Name]                                                                                  |
| Cédula                       | [ID Number]                                                                                  |
| Teléfono                     | [Phone Number]                                                                              
| Correo                       | [Email]                                                                                      
| Municipio                    | [Location]                                                                                   
| Asunto                       | [PQRS Description]                                                                          
| Dirección Asignada           | [Relevant CAR Direction based on the subject]                                                 |
| Justificación                | [Brief explanation of why this direction was selected]                                         |
| Tipo de Respuesta            | RESPUESTA A OFICIO                                                                            |
| Tipo Remitente               | [Juridica, Natural, Anonima]                                                                  |
| Fecha                        | [Date identified in the text]                                                                  |
| Proceso especial             | [No aplica, Thoman Van der Hammen, Rios Bogota, Cerros Orientales, Auditorias, Entes de Control, DRMI Fuquene, Reporte de Licencia de parcelacion y construccion, Proceso Eleccion Rep. Sector Privado] |
| Tipo de Tramite              | [Acciones Constitucionales, Certificación Ambiental para propuesta de Concesión Minera, Curadurías, DP Congreso de la República Ley 5/92 10 días, DP Congreso de la República Ley 5/92 48h, DP Congreso de la República Ley 5/92 5 días, Dp de Consulta, Dp de interés Particular, Dp, de oficio Permisivos, Dp, Defensoria del Pueblo Ley 5/92 5 días, Dp En cumplimiento de un deber legal (permisos, DP permisivos, Dp queja Ambiental (Afectación ambiental), Dp queja por atención al servicio)] |
| Departamento                  | [Department Name]                                                                              |
| Vereda                       | [If applicable, name of the village]                                                          |
| Predio                       | [If the property name is provided, include it]                                                |
| Medio de documento           | Oficio                                                                                        
| Numero de Folios             | 1                                                                                            
| Anexos                        | EMPTY                                                                                         
| Observaciones                | [Summary of what the person is asking in the PQRS]                                            |
| Copia a                      | EMPTY                                                                                         
| Quien Entrega                | [Empresa de mensajería, Persona Natural]                                                       |
| Atención Preferencial        | [Aulto Mayor, Desplazado (Víctimas de violencia/conflicto armado), Discapacidad física, Discapacidad Mental, Discapacidad Sensorial, Grupos Étnicos Minoritarios, Mujer Embarazada, Niños o Adolescentes, Periodista, Veterano de la Fuerza Pública] |


Rules for direction assignment:
1. Carefully analyze the subject matter of the PQRS
2. Select the most appropriate direction based on their competencies
3. Provide a brief justification for the assignment
4. If the subject involves multiple directions, select the primary one most relevant to the main issue

For regular conversation (no 'PQRS:' prefix), respond naturally as a helpful assistant with knowledge about CAR's structure and functions.
"""

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        # Process and display the response
        display_response(self.text, self.container)

def extract_table_data(markdown_text):
    """Extract table data from markdown and convert to DataFrame."""
    try:
        # Find table in text using regex
        table_pattern = r'\|.*\|'
        table_rows = re.findall(table_pattern, markdown_text)
        
        if not table_rows:
            return None, None
            
        # Process table rows
        headers = ['Campo', 'Valor']  # Fixed headers for consistent display
        data = []
        
        # Skip separator row (|---|---|)
        for row in table_rows[2:]:
            values = [col.strip() for col in row.split('|')[1:-1]]
            if len(values) == 2:  # Ensure we have both campo and valor
                data.append(values)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Extract non-table text
        pre_table = markdown_text.split('|')[0].strip()
        post_table = markdown_text.split('|')[-1].strip()
        other_text = f"{pre_table}\n\n{post_table}".strip()
        
        return df, other_text
    except Exception as e:
        st.error(f"Error processing table: {str(e)}")
        return None, None

def display_response(response_text, container):
    """Display the response using Streamlit components."""
    if '|' in response_text:  # Check if response contains a table
        df, other_text = extract_table_data(response_text)
        if df is not None:
            # Display any text before the table
            if other_text:
                container.markdown(other_text)
            
            # Display the DataFrame with enhanced styling
            container.markdown("### Información PQRS")
            
            # Apply custom styling to the DataFrame
            styled_df = df.style.set_properties(**{
                'background-color': '#f0f2f6',
                'color': '#1f1f1f',
                'border': '2px solid #add8e6'
            })
            
            # Display using st.dataframe with enhanced configuration
            container.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Campo": st.column_config.TextColumn(
                        "Campo",
                        help="Categoría de la información",
                        width="medium",
                    ),
                    "Valor": st.column_config.TextColumn(
                        "Valor",
                        help="Información proporcionada",
                        width="large",
                    )
                }
            )
        else:
            container.markdown(response_text)
    else:
        container.markdown(response_text)

def get_chat_response(prompt, model_choice="OpenAI", temperature=0.3):
    """Generate chat response using the selected LLM."""
    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # Initialize chat model with API key from environment
        if model_choice == "OpenAI":
            chat_model = ChatOpenAI(
                model="gpt-4o",
                temperature= temperature,
                api_key=API_KEY,
                streaming=True,
                callbacks=[stream_handler]
            )
        elif model_choice == "Groq API":
            chat_model = None  # Placeholder for Groq API model
            st.warning("Groq API model is not yet implemented.")
        elif model_choice == "Claude":
            chat_model = None  # Placeholder for Claude model
            st.warning("Claude model is not yet implemented.")
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        # Add context from previous messages
        if "messages" in st.session_state:
            for msg in st.session_state.messages[-3:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(SystemMessage(content=msg["content"]))
        
        response = chat_model.invoke(messages)
        return stream_handler.text
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Lo siento, ocurrió un error al procesar su solicitud."


def main():
    st.set_page_config(page_title="CARresponde", layout="centered")
    #st.write(logo, unsafe_allow_html=True)
    st.title("Cundi", anchor=False)
    st.markdown("**Soy Cundi, tú asistente virtual para la CAR. Entiende tus Peticiones, Quejas, Reclamos y Solicitudes (PQRS)**")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add a button to clear chat history
    # Add a button to clear chat history
    with st.sidebar:  
        st.markdown("""
**Bienvenido al Sistema de Gestión de PQRS**      
Esta herramienta está diseñada para ayudarte a clasificar y gestionar eficientemente las PQRS recibidas, puedes:    
                                   
- Descargar el desglose de la PQRS: Obtén un informe detallado de tus solicitudes.
- Haz clic en la casilla para ver más texto: Accede a información adicional sobre tu consulta.

**Para comenzar ingrea tu PQRS.**  
                """)
        
        model_choice = st.selectbox("Selecciona el modelo de IA deseado:", ["OpenAI", "Groq API", "Claude"])

        if st.button("Borra Historial del Chat"):
            st.session_state.messages = []
            st.experimental_rerun()            

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and '|' in message["content"]:
                # Process and display stored PQRS responses using DataFrame
                display_response(message["content"], st)
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Escribe tu mensaje acá... (Inicia con PQRS: para procesar el PQRS)"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("User",avatar="👨‍💼" ):
            st.markdown(prompt)
        
        # Process response
        with st.chat_message("ai", avatar="🌳"):
            is_pqrs = prompt.upper().startswith("PQRS:")
            if is_pqrs:
                pqrs_content = prompt[5:].strip()
                response = get_chat_response(pqrs_content)
            else:
                response = get_chat_response(prompt)
            
            # Store assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
