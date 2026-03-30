import chalk from "chalk"

import type {Version} from "./types.js"

export const supported_chromium_version: Version = [141, 0, 7390, 54]

export function get_version_tuple(version: string): Version | null {
  const match = version.match(/(\d+)\.(\d+)\.(\d+)\.(\d+)/)
  if (match != null) {
    const [, a, b, c, d] = match.values()
    return [
      Number(a),
      Number(b),
      Number(c),
      Number(d),
    ]
  }
  return null
}

export function check_version(current_chromium_version: Version | null): void {
  if (current_chromium_version == null) {
    const supported_str = supported_chromium_version.join(".")
    console.error(`${chalk.yellow("warning:")} unable to determine chromium version; officially supported version is ${supported_str}`)
    return
  }

  const [a, b, c, _d] = supported_chromium_version
  const [A, B, C, _D] = current_chromium_version

  if (a != A || b != B || c != C) {
    const supported_str = chalk.magenta(supported_chromium_version.join("."))
    const current_str = chalk.magenta(current_chromium_version.join("."))
    console.error(`${chalk.yellow("warning:")} ${current_str} is not supported; officially supported version is ${supported_str}`)
  }
}
