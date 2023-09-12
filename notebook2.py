from functools import partial

import jax
import jax.random as random
import jax.numpy as jnp
from jax import jit

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

#jax.config.update('jax_platform_name', 'cpu')

from typing import cast
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

ratings = jnp.arange(8)*100+1500
ids = jnp.arange(len(ratings))

key = random.PRNGKey(42)

def win_prob_bradley_terry(r1, r2):
    odds = 10.**((r2 - r1) / -400.)
    p = odds / (1. + odds)
    return p

def single_elimination(key, ids, ratings, rounds=3):
    for _ in range(rounds):
        key, subkey_perm, subkey_u = random.split(key, 3)
        idx = random.permutation(subkey_perm, len(ids))
        ids = ids[idx]
        ratings = ratings[idx]
        r1 = ratings[0::2]
        r2 = ratings[1::2]
        p = win_prob_bradley_terry(r1, r2)
        u = random.uniform(subkey_u, (len(p),))
        s = p > u
        idx = 2*jnp.arange(len(s)) + (1 - s)
        ids = ids[idx]
        ratings = ratings[idx]
    return ids

single_elimination(key, ids, ratings)

def single_elimination_rounds(key, ids, ratings, k):
    f = jax.vmap(single_elimination, (0, None, None))
    wins = f(random.split(key, k), ids, ratings)
    wins = wins.flatten()
    return jnp.argmax(jnp.bincount(wins, length=len(ids)))

n = 4000
k = 5
batched_rounds = jit(jax.vmap(partial(single_elimination_rounds, k=k), (0, None, None)))
champ = batched_rounds(random.split(key, n), ids, ratings)
champ_count = jnp.bincount(champ)
champ_probs = champ_count / jnp.sum(champ_count)

fig, ax = cast(tuple[plt.Figure, plt.Axes], plt.subplots())
ax.bar(jnp.arange(len(champ_probs)), champ_probs)
plt.savefig("single_elimination.pdf")
plt.close()

print(champ_probs[-1])



def random_comparison(key, ids, ratings):
    key, subkey = random.split(key)
    idx = random.choice(subkey, len(ids), shape=(2,), replace=False)
    r1, r2 = ratings[idx]
    p = win_prob_bradley_terry(r1, r2)
    key, subkey = random.split(key)
    u = random.uniform(subkey)
    s = p > u
    return ids[idx[1 - s]]

random_comparison(key, ids, ratings)

def random_comparison_rounds(key, ids, ratings, k):
    f = jax.vmap(random_comparison, (0, None, None))
    wins = f(random.split(key, k), ids, ratings)
    return jnp.argmax(jnp.bincount(wins, length=k))

n = 10000
k = 5*7 
batched_rounds = jit(jax.vmap(partial(random_comparison_rounds, k=k), (0, None, None)))
champ = batched_rounds(random.split(key, n), ids, ratings)
champ_count = jnp.bincount(champ)
champ_probs = champ_count / jnp.sum(champ_count)

fig, ax = cast(tuple[plt.Figure, plt.Axes], plt.subplots())
ax.bar(jnp.arange(len(champ_probs)), champ_probs)
plt.savefig("random_comparison.pdf")
plt.close()

print(champ_probs[-1])

import typing
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
ipython = typing.cast(InteractiveShell | None, get_ipython())

if ipython:
    ipython.run_line_magic("reload_ext", "autoreload")
    ipython.run_line_magic("autoreload", "1")

    ipython.run_line_magic("aimport", "glickbest_jax")

import glickbest_jax as glickbest

ratings_init = jnp.array([1500., 350.], dtype=jnp.float32)
glicko_ratings = jnp.tile(ratings_init, (8, 1))

def glickbest_rounds(key, ratings, true_ratings, k):
    def body(v, ratings):
        key_iter = random.fold_in(key, v)
        subkey_match, subkey_win = random.split(key_iter)
        i, j = glickbest.next_match(subkey_match, ratings)
        r1, r2 = true_ratings[i], true_ratings[j]
        p = win_prob_bradley_terry(r1, r2)
        u = random.uniform(subkey_win)
        s = p > u
        return glickbest.update(ratings, i, j, 1.*s)
    return jax.lax.fori_loop(0, k, body, init_val=ratings)

glickbest_rounds(key, glicko_ratings, ratings, k=3)

n = 10000
k = 5*7
batched_rounds = jit(jax.vmap(partial(glickbest_rounds, k=k), (0, 0, None)))
glicko_ratings_all = jnp.tile(ratings_init, (n, 8, 1))
out = batched_rounds(random.split(key, n), glicko_ratings_all, ratings)

champ_count = jnp.bincount(jnp.argmax(out[:,:,0], axis=1))
champ_probs = champ_count / jnp.sum(champ_count)

fig, ax = cast(tuple[plt.Figure, plt.Axes], plt.subplots())
ax.bar(jnp.arange(len(champ_probs)), champ_probs)
plt.savefig("glickbest.pdf")
plt.close()

print(champ_probs[-1])
