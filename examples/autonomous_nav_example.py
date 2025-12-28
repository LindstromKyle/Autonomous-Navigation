from autonomous_nav.app import AutonomousNavigationApp
from autonomous_nav.config import AppConfig

# Load the config
config = AppConfig()

# Initialize the application
app = AutonomousNavigationApp(config=config)

# Run
app.run()
