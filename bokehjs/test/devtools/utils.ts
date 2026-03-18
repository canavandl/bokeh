import {TimeoutError} from "./types.js"

const MAX_INT32 = 2147483647

export class Random {
  private seed: number

  constructor(seed: number) {
    this.seed = seed % MAX_INT32
    if (this.seed <= 0) {
      this.seed += MAX_INT32 - 1
    }
  }

  integer(): number {
    this.seed = (48271*this.seed) % MAX_INT32
    return this.seed
  }

  float(): number {
    return (this.integer() - 1) / (MAX_INT32 - 1)
  }
}

export function shuffle<T>(array: T[], random: Random): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(random.float()*(i + 1))
    const temp = array[i]
    array[i] = array[j]
    array[j] = temp
  }
}

export function timeout(ms: number): Promise<void> {
  return new Promise((_resolve, reject) => {
    const timer = setTimeout(() => reject(new TimeoutError()), ms)
    timer.unref()
  })
}

export function encode(s: string): string {
  return s.replace(/[ \/\[\]:]/g, "_")
}
