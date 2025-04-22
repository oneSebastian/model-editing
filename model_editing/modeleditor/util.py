from ..queryexecutor import QueryExecutor

class EditModel(QueryExecutor):
    '''
    If a model editor changes the model parameters these two methods (edit_model, restore_model) are sufficient.
    If instead if injects additional content into the prompt context,
      it also neads to implement the following methods for evaluation (compare to InContextModel):
      for lm eval, generate and options queries:
        - generate_until
        - loglikelihood
        - loglikelihood_rolling
      for argmax queries:
        - create_argmax_inputs
    '''
    def edit_model(self, facts):
        raise NotImplementedError()  # Override in concrete classes

    def restore_model(self):
        raise NotImplementedError()  # Override in concrete classes


class NoEditModel(EditModel):
    def edit_model(self, facts):
        pass

    def restore_model(self):
        pass
