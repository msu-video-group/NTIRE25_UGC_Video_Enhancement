import numpy as np
from scipy import optimize, stats
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def __LogLikelihood(data, hidden_vars):
  hidden_vars = np.exp(hidden_vars)

  choice_probabilities = np.divide(hidden_vars, np.add(hidden_vars, hidden_vars.transpose()))
  return np.sum(np.multiply(data, np.log(choice_probabilities)))


def __ConfidenceIntervals(data, hidden_vars, confidence):
  hidden_vars = hidden_vars.reshape([-1, 1])
  exp_sub = np.exp(hidden_vars - hidden_vars.transpose())
  term_deriv = np.divide(exp_sub, np.power(exp_sub + 1, 2)) * data
  hessian = term_deriv + term_deriv.transpose()
  diag = np.sum(term_deriv, axis=0) + np.sum(term_deriv, axis=1)
  hessian -= np.diag(diag)

  try:
    cov = np.linalg.inv(-hessian[1:, 1:])
  except np.linalg.linalg.LinAlgError:
    cov = np.zeros((hidden_vars.size - 1, hidden_vars.size - 1)) * np.nan

  cov_all = np.pad(cov, ((1, 0), (1, 0)), 'constant')
  all_relative_disp = np.diag(cov_all)[np.newaxis, :] + np.diag(cov_all)[:, np.newaxis] - 2 * cov_all

  trust_level = stats.norm.ppf((confidence + 1) / 2)
  bound = trust_level * all_relative_disp**0.5

  return bound, cov_all


def ComputeBradleyTerryRanksDense(wins, confidence=0.95, regularization=0.1):

  np.random.seed(seed=42)

  wins = wins + (np.ones(wins.shape) - np.eye(*wins.shape)) * regularization

  # Assume that ranks of the first method is `0`. And compute inverted log likelihood.
  inverted_likelihood = lambda ranks: __LogLikelihood(wins, np.vstack(([[0]], ranks.reshape([-1, 1])))) * -1

  # Initial guess for the rest of ranks.
  ranks_0 = np.zeros(wins.shape[0] - 1)

  result = optimize.minimize(inverted_likelihood, ranks_0, method='SLSQP')
  if not result.success:
    return None, None, None, None

  # Add rank of the first method to the list.
  ranks = np.vstack(([[0]], result.x.reshape([-1, 1])))
  ranks = ranks - np.min(ranks)
  confidence_intervals, cov = __ConfidenceIntervals(wins, ranks, confidence)

  return ranks, confidence_intervals, cov


def QuestionsToWins(questions, methods):
  methods = dict([reversed(m) for m in enumerate(methods)])
  wins = np.zeros([len(methods), len(methods)])

  for question in questions:
    left = methods[question['left_method']]
    right = methods[question['right_method']]
    answer = methods[question['answer']] if question['answer'] is not None else None
    if left == answer:
      wins[left, right] += 1
    elif right == answer:
      wins[right, left] += 1
    else:
      # Tie case.
      wins[left, right] += 0.5
      wins[right, left] += 0.5

  return wins


def ComputeBradleyTerryRanks(questions, methods, confidence=0.95):
  '''Computes methods ranks with Bradley-Terry model.

  Args:
      questions: iterable of dictionaries. Each dictionary corresponds to
          answered question and contains `left_method`, `right_method` and
          `answer` (i.e. the choosen method).
      methods: the list of method names/ids.
      confidence: confidence level for confidence interval computation.

  Returns:
      Dictionary holding for each method name/id its inferred rank.
  '''
  wins = QuestionsToWins(questions, methods)
  ranks, confidence_intervals, cov = ComputeBradleyTerryRanksDense(wins, confidence)
  if ranks is None:
    return None

  return {
      method: {
          'mean': mean,
          'confidence': {
              meth: (conf if not np.isnan(conf) else None)
              for meth, conf in zip(methods, conf_ints)
          },
          'cov': {meth: (cov_val if not np.isnan(cov_val) else None)
                  for meth, cov_val in zip(methods, mcov)}
      }
      for method, mean, conf_ints, mcov in zip(methods, ranks.flatten(), confidence_intervals, cov)
  }




def SubjectiveScores(votes_df, save_path = None):
    questions = []
    df = votes_df.copy(deep=True)
    df = df.where(pd.notnull(df), None)
    
    for idx, row in df.iterrows():
        questions.append({'left_method': row['left_method'],
                           'right_method': row['right_method'],
                           'answer': row['answer']})
    
    methods = list(set(df['left_method']).union(set(df['right_method'])))
    
    print("Running Bradley-Terry")
    print("Questions:", len(questions))
    print("Sequences:", len(set(df['test_case'])))
    print("Methods:", len(methods))
    
    ranks = ComputeBradleyTerryRanks(questions, methods)

    rows = [{'name': key,
             'subjective_score': str(ranks[key]['mean']),
             'confidence_95_to_original': str(ranks[key]['confidence']['original'])} for key in ranks.keys()]

    subjective_results = pd.DataFrame(rows, columns=['name', 
                                                     'subjective_score',
                                                     'confidence_95_to_original']).sort_values(by='subjective_score').reset_index(drop=True)

    if save_path is not None:
      subjective_results.to_csv(save_path, index=False)

    return subjective_results