import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // 변수 미사용 관련
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": "off",

      // any 관련
      "@typescript-eslint/no-explicit-any": "off",

      // useEffect/useMemo 의존성 검사
      "react-hooks/exhaustive-deps": "off",

      // <img> 태그 사용
      "@next/next/no-img-element": "off",

      // unescaped entity(따옴표, 쌍따옴표 등 특수문자)
      "react/no-unescaped-entities": "off",

      // 타입스크립트의 explicit function return type (혹시 추가로 떠 있다면)
      "@typescript-eslint/explicit-function-return-type": "off",

      // eslint-disable directive 미사용 경고
      "eslint-comments/no-unused-disable": "off",

      // 기타 자주 등장하는 react hook lint
      "react-hooks/rules-of-hooks": "off",
    },
  },
];

export default eslintConfig;
