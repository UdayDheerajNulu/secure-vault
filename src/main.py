import cv2
from deepface import DeepFace
from ultralytics import YOLO
from crewai import Agent, Task, Crew


# Load YOLOv8 model for CCTV monitoring
yolo_model = YOLO('yolov8n.pt')  # Pretrained model

# Load known face encodings (pre-saved authorized users)
authorized_faces = {
    "John Doe": "john_doe.png",
    "Alice Smith": "alice_smith.png"
}

def authenticate_face(frame):
    """Check if the detected face is in the authorized list."""
    try:
        result = DeepFace.find(frame, db_path="authorized_faces/")
        if len(result) > 0:
            return True, result[0]['identity']
    except:
        return False, None
    return False, None

def detect_intrusion(frame):
    """Detect unauthorized persons or unusual activity."""
    results = yolo_model(frame)
    person_count = 0
    unauthorized_objects = []
    
    for result in results:
        for box in result.boxes:
            detected_class = result.names[int(box.cls)]  # Get detected object class
            if detected_class == "person":
                person_count += 1
            elif detected_class in ["gun", "knife", "weapon"]:
                unauthorized_objects.append(detected_class)
    
    if person_count != 2 or unauthorized_objects:
        return True  # Intrusion detected if person count is not 2 or weapons are present
    return False

# Define an AI Security Agent
security_agent = Agent(
    role="Security AI",
    goal="Monitor vault security and take necessary action.",
    tools=[authenticate_face, detect_intrusion]
)

def security_task():
    """Task for monitoring vault security."""
    cap = cv2.VideoCapture(0)  # Capture from webcam/CCTV
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check for unauthorized access
        authorized, person = authenticate_face(frame)
        intrusion = detect_intrusion(frame)
        
        if not authorized and intrusion:
            print("\ud83d\udea8 Unauthorized person detected or security violation! Triggering alarm...")
        elif authorized:
            print(f"✅ Access granted to {person}")
        
        cv2.imshow("Vault Security Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Assign task to AI Agent
task = Task(description="Monitor vault entry and detect intrusions.", agent=security_agent)

# CrewAI to coordinate the security system
crew = Crew(agents=[security_agent], tasks=[task])
crew.kickoff()
