import sys
import os

# Root directory ko path me add karna
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashbord import server as application

app = application
