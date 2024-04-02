print_state([A,B,C,D,E,F,G,H,I]) :- 
    maplist(write, [A,B,C,"\n",D,E,F,"\n",G,H,I]), nl.

seq([X,Y,Z,_,_,_,_,_,_], [X,Y,Z], [0,1,2]).
seq([_,_,_,X,Y,Z,_,_,_], [X,Y,Z], [3,4,5]).
seq([_,_,_,_,_,_,X,Y,Z], [X,Y,Z], [6,7,8]).
seq([X,_,_,Y,_,_,Z,_,_], [X,Y,Z], [0,3,6]).
seq([_,X,_,_,Y,_,_,Z,_], [X,Y,Z], [1,4,7]).
seq([_,_,X,_,_,Y,_,_,Z], [X,Y,Z], [2,5,8]).
seq([X,_,_,_,Y,_,_,_,Z], [X,Y,Z], [0,4,8]).
seq([_,_,X,_,Y,_,Z,_,_], [X,Y,Z], [2,4,6]).

% valid state
x_o_diff([],0).
x_o_diff([x|T],Count) :- x_o_diff(T,CountPrev), Count is CountPrev + 1.
x_o_diff([o|T],Count) :- x_o_diff(T,CountPrev), Count is CountPrev - 1.
x_o_diff([b|T],Count) :- x_o_diff(T,Count).

valid_state(S) :- valid_elems(S), x_o_diff(S,0), \+ in_row_3(S).
valid_elems([A,B,C,D,E,F,G,H,I]) :- valid_elem(A), valid_elem(B), valid_elem(C), valid_elem(D), valid_elem(E),
    valid_elem(F), valid_elem(G), valid_elem(H), valid_elem(I).
in_row_3(S) :- seq(S, [X,Y,Z], _), X=o, Y=o,Z=o.
in_row_3(S) :- seq(S, [X,Y,Z], _), X=x, Y=x,Z=x.

valid_elem(E) :- E=o.
valid_elem(E) :- E=b.
valid_elem(E) :- E=x.


two_ways_x(S) :-
    valid_state(S),
    findall(Pos, (seq(S, Seq, Pos), winnable_for_x(Seq)), WinPositions),  % replace w bagof
    length(WinPositions, Count),
    Count >= 2, % assert at least 2 winning states
    print_state(S).


% if 2x ^ 1b in any order
winnable_for_x([x,x,b]).
winnable_for_x([x,b,x]).
winnable_for_x([b,x,x]).


no_ways_x(S) :-
    findall(Seq, seq(S, Seq, _), Seqs),
    % no sequence allows X to win in the next move
    maplist(no_immediate_win_x, Seqs).

no_immediate_win_x(Seq) :-
    count_elems(Seq, x, CountX),
    count_elems(Seq, b, CountB),
    %never finds immediate 2 x and 1 b? 
    \+ (CountX = 2, CountB = 1).

count_elems(List, Elem, Count) :-
    include(==(Elem), List, Filtered),
    length(Filtered, Count).
