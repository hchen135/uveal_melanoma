from flask import Flask

app = Flask(__name__)

import user_study.home
import user_study.task

# class 1: 
# correctly-classified: 24 (744, few cells, not confident), 13 (10596, lots of cells, confident)
# misclassified: 29 (1350, mix, follow second rule)
# class 2: 
# correctly-classified: 51 (861, few cells, confident, follow first rule), 59 (52279,lots of cells, confident, follow both rule, recommend second)
# misclassified: 65 (25784, not confident)

# thumbnail size:
# Slide 13: (549, 930)
# Slide 24: (555, 1116)
# Slide 29: (552, 987)
# Slide 51: (558, 1089)
# Slide 59: (561, 810)
# Slide 65: (555,906)

