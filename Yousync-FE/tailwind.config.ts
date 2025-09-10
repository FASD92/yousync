export const COLORS = {
  bg: "#101112",
  card: "#181a1b",
  border: "#232626",
  text: "#eaeaea",
  subText: "#929ca5",
  point: "#22ff88",
  pass: "#22ff88",
  fail: "#ee5566",
  white: "#fff",
  yellow: "#ffd56a"
};

export function getRankColor(rank: string) {
  if (rank === "S") return COLORS.point;
  if (rank === "A") return COLORS.yellow;
  if (rank === "B") return "#6cd4ff";
  return COLORS.fail;
}