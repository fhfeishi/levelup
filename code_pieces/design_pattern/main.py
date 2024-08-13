from singleton_once import model_zoo as init_once

from singleton_import import  module_

from singleton_decorator import Model_zoo as decorator_

from singleton_class import model_zoo as threading_

from singleton_meta import ModelZoo as meta_

# --method init_once
model_init_once = init_once()
print(model_init_once)

model_module_ = module_
print(model_module_)

model_decorator_ = decorator_()
print(model_decorator_) 

model_threading_ = threading_()
print(model_threading_)

model_meta_ = meta_()
print(model_meta_)