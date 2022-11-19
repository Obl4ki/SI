use itertools::Itertools;

#[derive(Clone, Debug)]
struct History {
    steps: Vec<Board>,
}

impl History {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Iterate through the states chronologically.
    fn iter(&self) -> impl Iterator<Item = &Board> {
        // Because the algorithm is recursive, history needs to be iterated backwards to be chronologically accurate
        self.steps.iter().rev()
    }

    fn add(&mut self, board: Board) {
        self.steps.push(board)
    }

    fn len(&self) -> usize {
        self.steps.len()
    }
}

fn main() {
    let start_state = vec![
        vec![PlayerMark::N, PlayerMark::O, PlayerMark::O],
        vec![PlayerMark::N, PlayerMark::X, PlayerMark::N],
        vec![PlayerMark::X, PlayerMark::N, PlayerMark::N],
    ];

    let (_, history) = start_as_x(start_state);
    print_history(&history);
    println!("len: {}", history.len());
}

type Board = Vec<Vec<PlayerMark>>;

#[allow(dead_code)]
fn start_as_x(start_state: Board) -> (f32, History) {
    min_max_with_alpha_beta_pruning(
        start_state,
        -1.,
        1.,
        History::new(),
        f32::min,
        f32::max,
        PlayerMark::X,
        PlayerMark::O,
    )
}

#[allow(dead_code)]
fn start_as_o(start_state: Board) -> (f32, History) {
    min_max_with_alpha_beta_pruning(
        start_state,
        -1.,
        1.,
        History::new(),
        f32::max,
        f32::min,
        PlayerMark::O,
        PlayerMark::X,
    )
}

/// Recursive implementation of tic-tac-toe with alpha-beta pruning.
#[allow(clippy::too_many_arguments)]
fn min_max_with_alpha_beta_pruning(
    node: Board,
    mut alpha: f32,
    mut beta: f32,
    mut history: History,
    comp_self: fn(f32, f32) -> f32,
    comp_other: fn(f32, f32) -> f32,
    mark_self: PlayerMark,
    mark_other: PlayerMark,
) -> (f32, History) {
    let end_winner = game_score(&node);
    if end_winner != PlayerMark::N {
        // history.add(node, ProcessingState::Pending);
        let v = if end_winner == PlayerMark::X { -1. } else { 1. };
        history.add(node);
        return (v, history);
    }
    let mut v = comp_self(-f32::INFINITY, f32::INFINITY);

    for child in get_children(&node, mark_self) {
        let (descendant_v, mut h) = min_max_with_alpha_beta_pruning(
            child,
            alpha,
            beta,
            history.clone(),
            comp_other,
            comp_self,
            mark_other,
            mark_self,
        );

        v = comp_self(descendant_v, v);

        match mark_self {
            PlayerMark::X => beta = comp_self(beta, v),
            PlayerMark::O => alpha = comp_self(alpha, v),
            _ => {}
        }

        if alpha >= beta {
            h.add(node);
            return (v, h);
        }
    }
    history.add(node);

    (v, history)
}

// fn max_value(node: Board, alpha: f32, beta: f32, mut history: History) -> (f32, History) {
//     history.add(node.clone(), ProcessingState::Pending);

//     let end_winner = game_score(&node);
//     if end_winner != PlayerMark::N {
//         let v = match end_winner {
//             PlayerMark::X => -1.,
//             PlayerMark::O => 1.,
//             PlayerMark::N => 0.,
//         };

//         return (v, history);
//     }
//     let mut v = -f32::INFINITY;
//     for child in get_children(&node, PlayerMark::O) {
//         let (descendant_v, mut h) = min_value(child, alpha, beta, history.clone());

//         v = f32::max(descendant_v, v);
//         let alpha = f32::max(alpha, v);
//         if alpha >= beta {
//             h.add(node.clone(), ProcessingState::AlphaBetaShortcut(AlphaBetaState { alpha, beta, v }));

//             return (v, h);
//         }
//     }

//     (v, history)
// }

// fn min_value(node: Board, alpha: f32, beta: f32, mut history: History) -> (f32, History) {
//     history.add(node.clone(), ProcessingState::Pending);

//     let end_winner = game_score(&node);
//     if end_winner != PlayerMark::N {
//         let v = match end_winner {
//             PlayerMark::X => -1.,
//             PlayerMark::O => 1.,
//             PlayerMark::N => 0.,
//         };

//         return (v, history);
//     }
//     let mut v = f32::INFINITY;
//     for child in get_children(&node, PlayerMark::X) {
//         let (descendant_v, mut h) = max_value(child, alpha, beta, history.clone());
//         v = f32::min(descendant_v, v);
//         let beta = f32::min(beta, v);

//         if alpha >= beta {
//             h.add(node.clone(), ProcessingState::AlphaBetaShortcut(AlphaBetaState { alpha, beta, v }));

//             // println!("Alpha beta shortcut: {alpha} {beta} {v}");
//             return (v, h);
//         }
//     }
//     (v, history)
// }

fn get_children(node: &Board, player_mark: PlayerMark) -> Vec<Board> {
    let mut children = vec![];
    for (x, y) in (0..3).cartesian_product(0..3) {
        if node[x][y] == PlayerMark::N {
            let mut new_state = node.clone();
            new_state[x][y] = player_mark;
            children.push(new_state);
        }
    }
    children
}

#[derive(PartialEq, Copy, Clone, Debug)]
enum PlayerMark {
    X,
    O,
    N,
}

fn game_score(node: &Board) -> PlayerMark {
    if node[0][0] == node[1][1] && node[1][1] == node[2][2] && node[0][0] != PlayerMark::N {
        return node[0][0];
    }
    if node[0][2] == node[1][1] && node[1][1] == node[2][0] && node[0][2] != PlayerMark::N {
        return node[0][2];
    }
    for i in 0..3 {
        if node[i][0] == node[i][1] && node[i][1] == node[i][2] && node[i][0] != PlayerMark::N {
            return node[i][0];
        }
        if node[0][i] == node[1][i] && node[1][i] == node[2][i] && node[0][i] != PlayerMark::N {
            return node[0][i];
        }
    }
    PlayerMark::N
}

fn print_history(history: &History) {
    for (idx, board) in history.iter().enumerate() {
        println!("idx: {}", idx);
        print_board(board);
    }
}

fn print_board(board: &Board) {
    for row in board {
        println!("{:?}", row);
    }
    println!();
}
