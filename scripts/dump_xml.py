#!/usr/bin/env python3
import sys
sys.path.insert(0,'.')
from src.envs.hns_environment import TeamCosEnv
from src.envs.env_xml_builder import EnvXMLBuilder
env=TeamCosEnv()
print(EnvXMLBuilder(env).build_dynamic_xml())
